"""把地瓜预编译的 llm 交互式二进制（纯文本 Qwen3-8B）包成常驻进程。

给 planner（大脑）用：cache_4096 上下文。llm 二进制是 REPL，单行 getline 读输入，
所以送进去的 prompt 必须是单行（FC 仿真在 openai_fc 里把多轮+tools 拍平成单行）。
协议与 vlm 一致：[Assistant] >>> 回答；下一轮 [User] <<< 作哨兵。
"""

import json
import os
import re
import select
import subprocess
import threading
import time

from . import config

PROMPT_SENTINEL = "[User] <<< "
ASSISTANT_MARK = "[Assistant] >>> "
PERF_MARK = "[Performance]"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class LlmWorker:
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._buf = ""
        self._buf_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._ready = False

    def start(self):
        config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        cfg_path = config.RUNTIME_DIR / "qwen3_8b.json"
        cfg_path.write_text(json.dumps(config.LLM_CONFIG, ensure_ascii=False, indent=2))
        env = dict(os.environ)
        env.update(config.HBM_ENV)
        self._proc = subprocess.Popen(
            [str(config.LLM_BIN), "-c", str(cfg_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(config.LLM_BIN.parent), env=env, bufsize=0,
        )
        threading.Thread(target=self._read_loop, daemon=True).start()
        self._wait_for(PROMPT_SENTINEL, timeout=120)
        self._ready = True

    def _read_loop(self):
        fd = self._proc.stdout.fileno()
        while True:
            r, _, _ = select.select([fd], [], [], 1.0)
            if r:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                with self._buf_lock:
                    self._buf += chunk.decode("utf-8", errors="replace")
            if self._proc.poll() is not None:
                break

    def _drain(self):
        with self._buf_lock:
            self._buf = ""

    def _wait_for(self, needle: str, timeout: float) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._buf_lock:
                idx = self._buf.find(needle)
                if idx != -1:
                    captured = self._buf[: idx + len(needle)]
                    self._buf = self._buf[idx + len(needle):]
                    return captured
            time.sleep(0.02)
        raise TimeoutError(f"等待记号超时: {needle!r}")

    def _send(self, line: str):
        self._proc.stdin.write((line + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def generate(self, prompt: str, timeout: float = 180) -> str:
        if not self._ready or self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("llm 进程未就绪")
        # prompt 必须单行（REPL 单行读取）；enable_multi_turn=false 时每轮自动 new_chat，无需 reset
        prompt = prompt.replace("\r", " ").replace("\n", " ")
        with self._call_lock:
            self._drain()
            self._send(prompt)
            # 推理是 async，结束时回调打印 [Performance] 再打印 [User] <<<，用前者判完成更稳
            captured = self._wait_for(PERF_MARK, timeout=timeout)
        return self._extract(captured)

    @staticmethod
    def _extract(raw: str) -> str:
        text = ANSI_RE.sub("", raw)
        if ASSISTANT_MARK in text:
            text = text.split(ASSISTANT_MARK, 1)[1]
        for cut in (PERF_MARK, PROMPT_SENTINEL):
            if cut in text:
                text = text.split(cut, 1)[0]
        return text.strip()

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._send("exit")
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
