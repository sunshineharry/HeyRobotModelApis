"""把地瓜预编译的 whisper 二进制包成一次性转写调用。

whisper 二进制是一锤子买卖：whisper --config_path cfg --audio_path x.wav
跑完打印 "[Transcription] <文字>" 和性能行后退出。
输入要求 16k 采样率的 wav；不是就先用 ffmpeg 转一下。
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TRANS_MARK = "[Transcription]"


def _ensure_wav16k(src: Path) -> tuple[Path, Path | None]:
    """返回 (可用的 wav 路径, 需要清理的临时目录或 None)。"""
    if src.suffix.lower() == ".wav":
        return src, None
    if not shutil.which("ffmpeg"):
        raise RuntimeError("输入非 wav 且板上没有 ffmpeg，无法转码")
    tmpdir = Path(tempfile.mkdtemp(prefix="asr_", dir=config.RUNTIME_DIR))
    out = tmpdir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(out)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out, tmpdir


def transcribe(audio_path: str, language: str | None = None) -> str:
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(config.WHISPER_CONFIG)
    if language:
        cfg["language"] = language
    cfg_path = config.RUNTIME_DIR / "whisper.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    wav, tmpdir = _ensure_wav16k(Path(audio_path))
    env = dict(os.environ)
    env.update(config.HBM_ENV)
    try:
        proc = subprocess.run(
            [str(config.WHISPER_BIN), "--config_path", str(cfg_path), "--audio_path", str(wav)],
            cwd=str(config.WHISPER_BIN.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    out = ANSI_RE.sub("", proc.stdout.decode("utf-8", errors="replace"))
    if TRANS_MARK not in out:
        raise RuntimeError(f"whisper 没有产出转写结果：\n{out[-800:]}")
    text = out.split(TRANS_MARK, 1)[1]
    text = text.split("[Performance]", 1)[0]
    return text.strip()
