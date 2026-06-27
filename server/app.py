"""OpenAI 兼容的本地大模型服务（S600 端侧，四件套）。

只服务**感知/语音/检测**这类端侧模型，方便 Xbotics-Hey-Robot 这类只认
OpenAI 协议的客户端把 base_url 指到本地就用上：
  - POST /v1/chat/completions      -> Qwen3-VL-4B 视觉（messages 必须带 image_url）
  - POST /v1/audio/transcriptions  -> whisper-medium（上传音频转文字）
  - POST /v1/audio/speech          -> WeTTS 合成
  - POST /v1/detect                -> YOLO26x 检测
  - GET  /v1/models                -> 列出可用模型

大脑 planner 不在本服务内：S600 部署 v1 让 planner 走线上 OpenAI 兼容大模型，
本服务不再承载本地 LLM planner。
VLM 进程在启动时常驻加载；whisper 每次请求起一个一次性进程。
"""

import base64
import binascii
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from . import asr_worker, config
from .detect_worker import DetectWorker
from .tts_worker import TtsWorker
from .vlm_worker import VlmWorker

vlm = VlmWorker()      # 视觉 Qwen3-VL-4B
tts = TtsWorker()
detector = DetectWorker()  # 懒加载：首次 /v1/detect 才占 BPU


@asynccontextmanager
async def lifespan(app: FastAPI):
    vlm.start()
    tts.start()
    yield
    vlm.stop()
    tts.stop()
    detector.stop()


app = FastAPI(title="S600 端侧大模型服务", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    now = int(time.time())
    data = [
        {"id": config.VLM_MODEL_ID, "object": "model", "created": now, "owned_by": "d-robotics"},
        {"id": config.WHISPER_MODEL_ID, "object": "model", "created": now, "owned_by": "d-robotics"},
        {"id": config.TTS_MODEL_ID, "object": "model", "created": now, "owned_by": "wetts"},
        {"id": config.DETECT_MODEL_ID, "object": "model", "created": now, "owned_by": "d-robotics"},
    ]
    return {"object": "list", "data": data}


def _save_image(url: str) -> str:
    """把 image_url 里的图存成临时文件，返回路径。支持 data URI 与 http(s)。"""
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(suffix=".jpg", dir=config.RUNTIME_DIR, delete=False)
    if url.startswith("data:"):
        try:
            b64 = url.split(",", 1)[1]
            fd.write(base64.b64decode(b64))
        except (IndexError, binascii.Error) as e:
            raise HTTPException(400, f"image_url 的 data URI 解析失败: {e}")
    elif url.startswith(("http://", "https://")):
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
        fd.write(resp.content)
    else:
        # 直接当本地路径
        fd.close()
        Path(fd.name).unlink(missing_ok=True)
        return url
    fd.close()
    return fd.name


def _parse_messages(messages: list[dict]) -> tuple[str, str | None]:
    """从 OpenAI messages 取出最后一条 user 的文本与（可选）图片路径。"""
    prompt_parts: list[str] = []
    image_path: str | None = None
    # system 提示拼到 prompt 前面，保证场景描述等指令生效
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                prompt_parts.append(content.strip())
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user is None:
        raise HTTPException(400, "messages 里没有 user 消息")
    content = last_user.get("content")
    if isinstance(content, str):
        prompt_parts.append(content)
    elif isinstance(content, list):
        for item in content:
            itype = item.get("type")
            if itype == "text":
                prompt_parts.append(item.get("text", ""))
            elif itype == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url:
                    image_path = _save_image(url)
    return "\n".join(p for p in prompt_parts if p).strip(), image_path


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    """仅用于视觉：messages 必须带 image_url，路由到本地 Qwen3-VL-4B。

    本服务不含本地 planner（v1 让 planner 走线上）；纯文本/带 tools 的请求请直接
    打到线上 OpenAI 兼容端点，不要发到这里。
    """
    messages = body.get("messages")
    if not messages:
        raise HTTPException(400, "缺少 messages")
    prompt, image_path = _parse_messages(messages)
    if not image_path:
        raise HTTPException(
            400,
            "本地 /v1/chat/completions 仅服务视觉（messages 需带 image_url）；"
            "planner 走线上 OpenAI 兼容端点，不在本服务内。",
        )
    try:
        content = vlm.generate(prompt or "请描述这张图片。", image_path)
    except Exception as e:
        raise HTTPException(500, f"VLM 推理失败: {e}")
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.VLM_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content or ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=config.WHISPER_MODEL_ID),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
):
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, dir=config.RUNTIME_DIR, delete=False)
    tmp.write(await file.read())
    tmp.close()
    try:
        text = asr_worker.transcribe(tmp.name, language=language)
    except Exception as e:
        raise HTTPException(500, f"ASR 转写失败: {e}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    if response_format == "text":
        return JSONResponse(text)
    return {"text": text}


@app.post("/v1/audio/speech")
async def speech(body: dict):
    text = (body.get("input") or "").strip()
    if not text:
        raise HTTPException(400, "缺少 input 文本")
    speaker_id = int(body.get("speaker_id", 0))
    try:
        wav = tts.synth_wav(text, speaker_id=speaker_id)
    except Exception as e:
        raise HTTPException(500, f"TTS 合成失败: {e}")
    # WeTTS 产出 wav；OpenAI 的 response_format 这里固定回 wav
    return Response(content=wav, media_type="audio/wav")


@app.post("/v1/detect")
async def detect(
    file: UploadFile = File(...),
    score_threshold: float = Form(default=0.35),
    person_only: bool = Form(default=True),
):
    """YOLO26（BPU）目标检测，默认只回 person 框，供人体跟踪等调用。"""
    data = await file.read()
    try:
        result = detector.detect(data, score_thres=score_threshold, person_only=person_only)
    except Exception as e:
        raise HTTPException(500, f"检测失败: {e}")
    return result
