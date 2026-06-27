# S600 端侧大模型本地服务

在 RDK S600 板上，把端侧模型包成一套 **OpenAI 兼容 HTTP 接口**，让 Xbotics-Hey-Robot 这类
只认 OpenAI 协议的客户端，把 `base_url` 指到本地就能把原来调云端的视觉/语音/语音合成换成
S600 端侧推理：

| 能力 | 模型 | 跑在 | 端点 |
|---|---|---|---|
| VLM 视觉问答 | Qwen3-VL-8B（地瓜 BPU hbm） | **BPU** | `POST /v1/chat/completions`（支持 image_url） |
| ASR 语音转写 | whisper-medium（地瓜 BPU hbm） | **BPU** | `POST /v1/audio/transcriptions` |
| TTS 语音合成 | WeTTS VITS（取自 hobot_tts，onnx） | **CPU** | `POST /v1/audio/speech` |

> S600 SDK 没有 BPU 版 TTS（`oellm_runtime` 只有 LLM/VLM/VLA/ASR），所以 TTS 走 CPU 的 WeTTS；
> 一份 Qwen3-VL-8B + whisper 可同时常驻 BPU（balanced 下 `ion_carveout` ≈10GB），TTS 在 CPU 不抢 BPU。

板端部署目录：`~/S600_models`（uv 管理，代码）；模型 hbm 在 `/mnt/models`（本地源，777）。
本目录是它在仓库里的同源副本（不含模型大文件）。

## 目录结构

```
~/S600_models/                  # uv 项目（代码）
├── pyproject.toml              # fastapi / uvicorn / httpx / python-multipart
├── run_server.sh
├── server/
│   ├── config.py               # 路径/模型文件名/HBM 环境变量（MODELS_DIR=/mnt/models）
│   ├── vlm_worker.py           # 常驻 vlm 进程（8B 热加载），喂图+prompt 取回答
│   ├── asr_worker.py           # 调 whisper 二进制一次性转写（非 wav 自动 ffmpeg 转 16k）
│   ├── tts_worker.py           # ctypes 常驻 WeTTS libtts.so，文本→PCM→wav
│   └── app.py                  # FastAPI：OpenAI 兼容端点
└── .runtime/                   # 运行期生成的 config json、临时图片/音频

/mnt/models/                    # 模型本地源（777）
├── Qwen3-VL-8B-Instruct/       # vision + language + embed（w4，~5.9G）
├── whisper-medium/             # encode + decode（w8，~1.3G）
└── wetts_tts/                  # WeTTS：lib/libtts.so + libonnxruntime + tts_model/（~210M）
```

服务不重写推理：VLM/ASR 直接调地瓜 `~/D-Robotics_LLM_S600_1.0.2_SDK/oellm_runtime/` 的预编译
二进制 `vlm`/`whisper`；TTS 通过 ctypes 调 WeTTS 的 `libtts.so`。

## 前置（板端一次性）

1. **ION 内存切 balanced**（8B 必需）。本板 `RDK S600 MCB V0p2`，官方 `hb_switch_ion.sh` 写死只改
   v0p1.dtb、对本板无效，需手动改 v0p2 的 dtb 再重启：
   ```bash
   DTB=/boot/hobot/rdk-s600-mcb-v0p2.dtb
   sudo cp $DTB ${DTB}.bak
   sudo fdtput -t x $DTB /reserved-memory/ion_reserved reg 0x40 0xC0000000 0x0 0x80000000
   sudo fdtput -t x $DTB /reserved-memory/ion_carveout reg 0x41 0x40000000 0x2 0x80000000
   sudo fdtput -t x $DTB /reserved-memory/ion_cma      reg 0x43 0xC0000000 0x0 0x80000000
   sudo fdtput -t x $DTB /reserved-memory/ion_uncache  reg 0x44 0x40000000 0x0 0x80000000
   sudo reboot   # 重启后 ion_carveout 应为 10GB（reg ... 2 80000000）
   ```
2. **模型就位**（见「模型来源」），放在 `/mnt/models/`。
3. **装 uv 并同步依赖**：
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   cd ~/S600_models && uv sync
   ```

## 启动 / 停止

systemd user service（已配 linger，开机自起、崩溃自拉）：

```bash
systemctl --user start  s600-models      # 首启加载 8B+whisper+WeTTS 约 15–20s
systemctl --user stop   s600-models
systemctl --user status s600-models
journalctl --user -u s600-models -f
```

临时前台调试：`cd ~/S600_models && ./run_server.sh`。服务监听 `0.0.0.0:8000`。

## 接口（OpenAI 兼容）

```bash
# VLM：messages 带 image_url（data: base64 或 http(s)）
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen3-VL-8B-Instruct",
  "messages":[{"role":"user","content":[
    {"type":"text","text":"这张图里有什么？"},
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,...."}}]}]}'

# ASR：multipart 上传音频（非 16k wav 自动转码）
curl -s http://localhost:8000/v1/audio/transcriptions -F file=@audio.wav -F language=zh
# -> {"text":"..."}

# TTS：input 文本 -> 返回 wav（16kHz mono）
curl -s http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"wetts-vits-zh","input":"你好，我是本地语音助手。"}' -o out.wav

# 列出模型
curl -s http://localhost:8000/v1/models
```

实测：VLM 8B 首次加载 ~14s（常驻后不再重复）、单次图文问答 ~1s 级；ASR/TTS 均秒级。

## Xbotics-Hey-Robot 切到本地（已完成）

Hey-Robot 的视觉/语音/语音合成全部改成调本地服务。改动提交在 fork：
**`sunshineharry/Xbotics-Hey-Robot` 分支 `feat/s600-local-providers`**（基于上游最新）。

- **VLM**（`scene_captioner`）：`configs/xlerobot.real.ubuntu.yaml` 由 `type: dashscope`（云 Qwen-VL）
  改为 `type: openai_compat` + `model: Qwen3-VL-8B-Instruct` + `api_base: http://127.0.0.1:8000/v1`。
  （所有 reasoning provider 最终走 `OpenAICompatReasoningProvider`，只换 base_url。）
- **ASR**：新增 `OpenAIASRClient`（`src/hey_robot/audio/asr.py`，遵循已有 `ASRClient` 协议，POST
  `/v1/audio/transcriptions`），工厂注册 provider `openai`；yaml `channels.voice.asr` 切到它。
- **TTS**：新增 `OpenAISpeechTTSClient` + `build_tts_client` 工厂（`audio/tts.py`，返回 raw PCM16
  与 doubao 客户端同形），`voice_loop` 改用工厂；yaml `tts` 由 `doubao`（云）切到 `openai` 本地。

冒烟（Hey-Robot 自己的 client 互跑）：TTSClient 合成 → ASRClient 转写，文本回环通过。

```bash
cd ~/Xbotics-Hey-Robot && uv run python - <<'PY'
import asyncio, wave, io
from hey_robot.audio.config import TTSConfig, ASRConfig
from hey_robot.audio.tts import build_tts_client
from hey_robot.audio.asr import build_asr_client
tts = build_tts_client(TTSConfig(provider="openai", endpoint="http://127.0.0.1:8000/v1", resource_id="wetts-vits-zh", sample_rate=16000))
asr = build_asr_client(ASRConfig(provider="openai", endpoint="http://127.0.0.1:8000/v1", model="whisper-medium", language="zh"))
async def main():
    pcm = await tts.synthesize("今天天气很好，我们一起去公园散步吧。")
    b = io.BytesIO(); w=wave.open(b,"wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm); w.close()
    print(await asr.transcribe_wav(b.getvalue()))
asyncio.run(main())
PY
```

> planner 仍是云端 DeepSeek（未动；本次只本地化视觉/语音/合成）。

## 模型来源 / 下载

一条命令把三类模型下到本地源目录（默认 `/mnt/models`，布局与 `server/config.py` 一致）：

```bash
./download_models.sh                 # 默认 /mnt/models
MODELS_DIR=~/models ./download_models.sh   # 自定义目录
```

它下载：
- **Qwen3-VL-8B-Instruct**（VLM，w4，1.0.2）：vision/language/embed 三件（地瓜 OSS）。
- **whisper-medium**（ASR，w8，1.0.0）：encode/decode 两件（地瓜 OSS）。
- **WeTTS**（TTS，CPU onnx）：`libtts.so`+`libonnxruntime` + `tts_model/`（取自 `hobot_tts` + `archive.d-robotics.cc`）。

VLM/whisper 的官方下载清单亦见解压后地瓜 SDK 的 `oellm_runtime/model/resolve_model_nash-p.md`。
