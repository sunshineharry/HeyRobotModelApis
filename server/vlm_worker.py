"""把地瓜预编译的 vlm 交互式二进制包成一个常驻进程。

vlm 二进制是个 REPL：启动后加载模型（约几秒），随后逐行读 stdin：
  - 普通文本           -> 当作 prompt 跑一次推理
  - "/image <path>"    -> 加载一张图片
  - "reset"            -> 清空对话上下文
  - "exit"             -> 退出
输出里用固定记号分隔：每轮回答前打印 "[Assistant] >>> "，回答结束后打印
性能行（"===== ... ====="），再打印下一轮的输入提示 "[User] <<< "。
我们就用 "[User] <<< " 作为一轮结束的哨兵，把 "[Assistant] >>>" 与性能行之间的文字取出来。

模型常驻、单卡串行，所以这里用一把锁保证同一时刻只有一轮推理。
"""

import os
import re
import select
import subprocess
import threading
import time

from . import config

PROMPT_SENTINEL = "[User] <<< "
ASSISTANT_MARK = "[Assistant] >>> "
PERF_MARK = "====="
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class VlmWorker:
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._buf = ""
        self._buf_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._ready = False

    def start(self):
        config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        cfg_path = config.RUNTIME_DIR / "qwen3vl_8b.json"
        import json

        cfg_path.write_text(json.dumps(config.VLM_CONFIG, ensure_ascii=False, indent=2))

        env = dict(os.environ)
        env.update(config.HBM_ENV)
        self._proc = subprocess.Popen(
            [str(config.VLM_BIN), "-c", str(cfg_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(config.VLM_BIN.parent),
            env=env,
            bufsize=0,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        # 等模型加载完、第一次出现输入提示
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

    def generate(self, prompt: str, image_path: str | None = None, timeout: float = 180) -> str:
        if not self._ready or self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("vlm 进程未就绪")
        with self._call_lock:
            # 每次请求独立上下文
            self._drain()
            self._send("reset")
            self._wait_for(PROMPT_SENTINEL, timeout=10)
            if image_path:
                self._drain()
                self._send(f"/image {image_path}")
                # 加载图片后会再次出现输入提示
                self._wait_for(PROMPT_SENTINEL, timeout=30)
            self._drain()
            self._send(prompt)
            captured = self._wait_for(PROMPT_SENTINEL, timeout=timeout)
        return self._extract(captured)

    @staticmethod
    def _extract(raw: str) -> str:
        text = ANSI_RE.sub("", raw)
        if ASSISTANT_MARK in text:
            text = text.split(ASSISTANT_MARK, 1)[1]
        # 截掉性能统计与下一轮提示
        for cut in (PERF_MARK, "[Performance]", PROMPT_SENTINEL):
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
