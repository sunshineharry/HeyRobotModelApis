# S600 端侧大模型本地服务（HeyRobotModelApis）

在 RDK S600 上，把端侧模型包成一套 **OpenAI 兼容 HTTP 接口**，让 Xbotics-Hey-Robot 这类只认
OpenAI 协议的客户端，把 `base_url` 指到本地，就能把**视觉/语音识别/语音合成/检测换成 S600 端侧推理**。

> **S600 部署 v1（planner 线上版）**：大脑 planner 不在本服务内——v1 让 planner 走**线上** OpenAI
> 兼容大模型；本服务只承载端侧的视觉/语音/检测**四件套**，不含本地 LLM planner。

| 角色 | 模型 | 跑在 | 端点 |
|---|---|---|---|
| 视觉 VLM（场景描述） | **Qwen3-VL-4B**（Qwen） | BPU | `POST /v1/chat/completions`（必须带 image_url） |
| 语音识别 ASR | **whisper-medium** | BPU | `POST /v1/audio/transcriptions` |
| 语音合成 TTS | **WeTTS VITS**（取自 hobot_tts） | CPU | `POST /v1/audio/speech` |
| 人体/目标检测 | **YOLO26x**（取自 yolo26x_demo） | BPU | `POST /v1/detect`（默认只回 person） |

`/v1/chat/completions` **仅服务视觉**：`messages` 必须带 `image_url` → Qwen3-VL-4B；纯文本/带 tools 的
planner 请求请直接打到线上端点，不要发到本服务。

## 关键：BPU 内存（carveout）

四件套要同时常驻/并发，10GB carveout 装不下（whisper 会 OOM）。**本部署用 12GB carveout**（实测
VL-4B + whisper + yolo + WeTTS 共存可用），CPU 仍剩约 12GB。视觉用 **4B 不用 8B**——8B 的 hbm 与
whisper 同驻会让 whisper `hbDNNInitializeFromFiles` 失败（BPU 装不下）。本板 `MCB V0p2`，官方
`hb_switch_ion.sh` 写死改 v0p1.dtb 无效，需手动改 v0p2：

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
    ├── vlm_worker.py    # 常驻 vlm 进程（Qwen3-VL-4B），喂图+prompt
    ├── asr_worker.py    # whisper 一次性转写（只吃 16k wav）
    ├── tts_worker.py    # ctypes 常驻 WeTTS libtts.so
    ├── detect_worker.py # 懒加载 in-process YOLO26x（系统 hbm_runtime+cv2，复用 yolo26x_demo runtime）
    └── app.py           # FastAPI：OpenAI 兼容端点
/mnt/models/                    # 模型本地源（777）
├── Qwen3-VL-4B-Instruct/  whisper-medium/  wetts_tts/  yolo26x_demo/
```

VLM/ASR 直接调地瓜 `oellm_runtime` 预编译二进制（vlm/whisper）；TTS ctypes 调 WeTTS；检测 in-process
调 hbm_runtime + yolo26x_demo 的 runtime。不重写推理、不改第三方库。

## 部署

```bash
# 1. ION 切 12GB carveout（见上）+ reboot
# 2. 装 uv + 下模型
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/sunshineharry/HeyRobotModelApis.git ~/S600_models && cd ~/S600_models
./download_models.sh          # 下 4 类模型到 /mnt/models
uv sync
# 3. 常驻服务（systemd user + linger，开机自起）
systemctl --user start s600-models     # 见 run_server.sh；首启加载 VL-4B+WeTTS ~30s
```

## API 端点一览

服务监听 `0.0.0.0:8000`。

| 方法 | 路径 | 角色/模型 | 请求 | 响应 |
|---|---|---|---|---|
| GET | `/health` | 健康检查 | — | `{"status":"ok"}` |
| GET | `/v1/models` | 列模型 | — | 4 个：Qwen3-VL-4B-Instruct / whisper-medium / wetts-vits-zh / yolo26x |
| POST | `/v1/chat/completions` | **视觉 VL-4B**（必须带图） | OpenAI Chat：`{model, messages}`，`messages` 须含 `image_url`（无图返回 400，planner 走线上） | OpenAI `chat.completion`：`choices[0].message.content` |
| POST | `/v1/audio/transcriptions` | **ASR whisper** | multipart：`file`(16k wav)、`model?`、`language?`(zh/en) | `{"text":"..."}` |
| POST | `/v1/audio/speech` | **TTS WeTTS** | JSON：`{input, model?}` | 16k mono 音频字节 |
| POST | `/v1/detect` | **检测 YOLO26x** | multipart：`file`(图)、`score_threshold?`(默认0.35)、`person_only?`(默认true) | `{width,height,detections:[{box:[x1,y1,x2,y2],score,class_id,class}]}` |

```bash
# 视觉（必须带图→VL-4B）
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-VL-4B-Instruct","messages":[{"role":"user","content":[{"type":"text","text":"看到了什么"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]}]}'
# ASR / TTS / 检测
curl -s localhost:8000/v1/audio/transcriptions -F file=@a.wav -F language=zh         # -> {"text":"..."}
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -d '{"input":"你好"}' -o out.audio
curl -s localhost:8000/v1/detect -F file=@frame.jpg                                  # -> {"detections":[{"box":[..],"score":..,"class":"person"}]}
curl -s localhost:8000/v1/models
```

## Xbotics-Hey-Robot 切到本地（opt-in，云端配置不动）

改动在 fork `sunshineharry/Xbotics-Hey-Robot` 分支 `feat/s600-local-providers`。
**原 `xlerobot.real.ubuntu.yaml`（云端）保持可用**；`xlerobot.real.s600.yaml` 把感知/语音指本地、planner 指线上：

- **planner（大脑）**：`type: openai_compat` + `model_env/api_base_env/api_key_env`（`PLANNER_LLM_MODEL / PLANNER_LLM_API_BASE / PLANNER_LLM_API_KEY`）——走**线上** OpenAI 兼容端点，端点/模型/key 由环境变量给、不写死配置。
- **scene_captioner（视觉）**：`model: Qwen3-VL-4B-Instruct` 指本服务。
- **ASR**：`OpenAIASRClient`，`channels.voice.asr.provider: openai` 指本地 whisper。
- **TTS**：`OpenAISpeechTTSClient`，`tts.provider: openai` 指本地 WeTTS。
- **人体跟随检测**：`perception/human_follow` opt-in，设 `S600_DETECT_URL=http://127.0.0.1:8000` 即走本地 BPU `/v1/detect`。

代码改动均为加法，云端 provider（dashscope/doubao/sherpa/deepseek）不受影响。

## 模型来源 / 下载

一条命令把 4 类模型下到 `/mnt/models`：`./download_models.sh`（VL/whisper 来自地瓜 OSS，见
`oellm_runtime/model/resolve_model_nash-p.md`；WeTTS 取自 hobot_tts；YOLO26x 取自 kol_test yolo26x_demo）。
