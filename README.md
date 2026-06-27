# S600 端侧大模型本地服务（HeyRobotModelApis）

在 RDK S600 上，把端侧模型包成一套 **OpenAI 兼容 HTTP 接口**，让 Xbotics-Hey-Robot 这类只认
OpenAI 协议的客户端，把 `base_url` 指到本地，就能把视觉/语音/语音合成/规划/检测**全部换成 S600
端侧推理，零云端依赖**。

| 角色 | 模型 | 跑在 | 端点 |
|---|---|---|---|
| 视觉 VLM（场景描述） | **Qwen3-VL-2B**（Qwen，视觉不需大模型） | BPU | `POST /v1/chat/completions`（带 image_url） |
| 大脑 planner（规划/工具调用） | **Qwen3-8B**（cache_4096 + 工具调用 FC 仿真） | BPU | `POST /v1/chat/completions`（带 tools，无图） |
| 语音识别 ASR | **whisper-medium** | BPU | `POST /v1/audio/transcriptions` |
| 语音合成 TTS | **WeTTS VITS**（取自 hobot_tts） | CPU | `POST /v1/audio/speech` |
| 人体/目标检测 | **YOLO26x**（取自 yolo26x_demo） | BPU | `POST /v1/detect`（默认只回 person） |

`/v1/chat/completions` 按是否带图自动路由：**带图→VL-2B 视觉**；**纯文本/带 tools→Qwen3-8B 大脑**。

## 关键：BPU 内存（carveout）

5 个模型要同时常驻/并发，10GB carveout 装不下（whisper 会 OOM、大模型互相干扰出乱码）。**本部署用
12GB carveout**（实测 VL-2B + Qwen3-8B + whisper + yolo 全并发跑通、5.3s、planner 正常出 tool_calls），
CPU 仍剩约 12GB。本板 `MCB V0p2`，官方 `hb_switch_ion.sh` 写死改 v0p1.dtb 无效，需手动改 v0p2：

```bash
DTB=/boot/hobot/rdk-s600-mcb-v0p2.dtb
sudo cp $DTB ${DTB}.bak
sudo fdtput -t x $DTB /reserved-memory/ion_reserved reg 0x40 0xC0000000 0x0 0x80000000
sudo fdtput -t x $DTB /reserved-memory/ion_carveout reg 0x41 0x40000000 0x3 0x00000000   # 12GB
sudo fdtput -t x $DTB /reserved-memory/ion_cma      reg 0x44 0x40000000 0x0 0x80000000
sudo fdtput -t x $DTB /reserved-memory/ion_uncache  reg 0x44 0xC0000000 0x0 0x80000000
sudo reboot   # 重启后 ion_carveout 应为 0x300000000(12GB)
```

## 目录结构

```
~/S600_models/                  # uv 项目（代码）
├── pyproject.toml  run_server.sh  download_models.sh
└── server/
    ├── config.py        # 路径/模型/HBM 环境变量（MODELS_DIR=/mnt/models）
    ├── vlm_worker.py    # 常驻 vlm 进程（Qwen3-VL-2B），喂图+prompt
    ├── llm_worker.py    # 常驻 llm 进程（Qwen3-8B），单行 prompt（planner）
    ├── openai_fc.py     # 工具调用 FC 仿真：tools→Qwen3 prompt(/no_think)，解析 <tool_call>→tool_calls
    ├── asr_worker.py    # whisper 一次性转写（非 wav 自动 ffmpeg 转 16k）
    ├── tts_worker.py    # ctypes 常驻 WeTTS libtts.so
    ├── detect_worker.py # 懒加载 in-process YOLO26x（系统 hbm_runtime+cv2，复用 yolo26x_demo runtime）
    └── app.py           # FastAPI：OpenAI 兼容端点

/mnt/models/                    # 模型本地源（777）
├── Qwen3-VL-2B-Instruct/  Qwen3-8B/  whisper-medium/  wetts_tts/  yolo26x_demo/
```

VLM/ASR/planner 直接调地瓜 `oellm_runtime` 预编译二进制（vlm/whisper/llm）；TTS ctypes 调 WeTTS；
检测 in-process 调 hbm_runtime + yolo26x_demo 的 runtime。不重写推理、不改第三方库。

## 部署

```bash
# 1. ION 切 12GB carveout（见上）+ reboot
# 2. 装 uv + 下模型
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/sunshineharry/HeyRobotModelApis.git ~/S600_models && cd ~/S600_models
./download_models.sh          # 下 5 类模型到 /mnt/models
uv sync
# 3. 常驻服务（systemd user + linger，开机自起）
systemctl --user start s600-models     # 见 run_server.sh；首启加载 VL-2B+Qwen3-8B+WeTTS ~30s
```

## API 端点一览

服务监听 `0.0.0.0:8000`。

| 方法 | 路径 | 角色/模型 | 请求 | 响应 |
|---|---|---|---|---|
| GET | `/health` | 健康检查 | — | `{"status":"ok"}` |
| GET | `/v1/models` | 列模型 | — | 5 个：Qwen3-VL-2B-Instruct / Qwen3-8B / whisper-medium / wetts-vits-zh / yolo26x |
| POST | `/v1/chat/completions` | **视觉 VL-2B**（带图）/ **大脑 Qwen3-8B**（纯文本或带 tools） | OpenAI Chat：`{model, messages, tools?, tool_choice?}`；`messages` 含 `image_url`→视觉，纯文本/带 `tools`→大脑 | OpenAI `chat.completion`：`choices[0].message.content`，或带 `tool_calls`（finish_reason=tool_calls） |
| POST | `/v1/audio/transcriptions` | **ASR whisper** | multipart：`file`(音频)、`model?`、`language?`(zh/en) | `{"text":"..."}` |
| POST | `/v1/audio/speech` | **TTS WeTTS** | JSON：`{input, model?}` | wav 字节（`audio/wav`，16k mono） |
| POST | `/v1/detect` | **检测 YOLO26x** | multipart：`file`(图)、`score_threshold?`(默认0.35)、`person_only?`(默认true) | `{width,height,detections:[{box:[x1,y1,x2,y2],score,class_id,class}]}` |

```bash
# 视觉（带图→VL-2B）
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-VL-2B-Instruct","messages":[{"role":"user","content":[{"type":"text","text":"看到了什么"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]}]}'
# 大脑/工具调用（带 tools→Qwen3-8B，回 tool_calls）
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-8B","messages":[{"role":"user","content":"查北京天气"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}'
# ASR / TTS / 检测
curl -s localhost:8000/v1/audio/transcriptions -F file=@a.wav -F language=zh         # -> {"text":"..."}
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -d '{"input":"你好"}' -o out.wav
curl -s localhost:8000/v1/detect -F file=@frame.jpg                                  # -> {"detections":[{"box":[..],"score":..,"class":"person"}]}
curl -s localhost:8000/v1/models
```

## Xbotics-Hey-Robot 切到本地（opt-in，云端配置不动）

改动在 fork `sunshineharry/Xbotics-Hey-Robot` 分支 `feat/s600-local-providers`（每功能一个 commit）。
**原 `xlerobot.real.ubuntu.yaml`（云端）保持可用**；新增 `xlerobot.real.s600.yaml` 全指本地：

- **planner**：`type: openai_compat`，`model: Qwen3-8B`，`api_base: http://127.0.0.1:8000/v1`（工具调用走本服务 FC 仿真）。
- **scene_captioner（视觉）**：`model: Qwen3-VL-2B-Instruct` 指本地。
- **ASR**：新增 `OpenAIASRClient`，`channels.voice.asr.provider: openai` 指本地 whisper。
- **TTS**：新增 `OpenAISpeechTTSClient`，`tts.provider: openai` 指本地 WeTTS。
- **人体跟随检测**：`perception/human_follow` 加 opt-in，设环境变量 `S600_DETECT_URL=http://127.0.0.1:8000` 即走本地 BPU `/v1/detect`，默认仍 ultralytics CPU。

代码改动均为加法，云端 provider（dashscope/doubao/sherpa/deepseek）不受影响。

## 模型来源 / 下载

一条命令把 5 类模型下到 `/mnt/models`：`./download_models.sh`（VL/whisper/Qwen3-8B 来自地瓜 OSS，见
`oellm_runtime/model/resolve_model_nash-p.md`；WeTTS 取自 hobot_tts；YOLO26x 取自 kol_test yolo26x_demo）。
