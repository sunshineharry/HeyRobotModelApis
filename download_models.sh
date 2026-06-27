#!/usr/bin/env bash
# 下载 S600 端侧模型到本地源目录（默认 /mnt/models），布局与 server/config.py 一致。
#   - Qwen3-VL-4B-Instruct   视觉 VLM（场景描述，BPU，w4）—— v1 planner 上云后视觉升 4B（8B 与 whisper 同驻装不下）
#   - whisper-medium         语音识别 ASR（BPU，w8）
#   - WeTTS                  语音合成 TTS（CPU onnx，取自 hobot_tts）
#   - yolo26x_demo           人体/目标检测（BPU，nash-p hbm + 官方 runtime）
# 用法：./download_models.sh           # 默认 /mnt/models
#       MODELS_DIR=~/models ./download_models.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/mnt/models}"
OSS="https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600"
HOBOT_TTS="https://github.com/D-Robotics/hobot_tts/raw/develop/wetts/lib"
WETTS_MODEL_URL="http://archive.d-robotics.cc/tts-model/tts_model.tar.gz"
YOLO26X_DEMO_URL="https://archive.d-robotics.cc/downloads/kol_test/yolo26x_demo.tar"

echo "[*] 模型目录: $MODELS_DIR"
mkdir -p "$MODELS_DIR" 2>/dev/null || sudo mkdir -p "$MODELS_DIR"
[ -w "$MODELS_DIR" ] || sudo chmod 777 "$MODELS_DIR"
dl() { wget -c -q --show-progress -O "$1" "$2"; }

# ---- 视觉 VLM: Qwen3-VL-4B-Instruct (w4, 1.0.0) ----
VLM="$MODELS_DIR/Qwen3-VL-4B-Instruct"; mkdir -p "$VLM"
echo "[*] Qwen3-VL-4B-Instruct (视觉) ..."
for f in \
  Qwen3-VL-4B-Instruct_vision_448x448_w8-4_nash-p_corenum_4.hbm \
  Qwen3-VL-4B-Instruct_language_chunk_512_cache_1024_w4_nash-p_corenum_4_4.hbm \
  Qwen3-VL-4B-Instruct_embed_tokens_w4_fp16.bin; do
  dl "$VLM/$f" "$OSS/1.0.0/models/Qwen3-VL-4B-Instruct/w4/$f"
done

# ---- ASR: whisper-medium (w8, 1.0.0) ----
ASR="$MODELS_DIR/whisper-medium"; mkdir -p "$ASR"
echo "[*] whisper-medium (ASR) ..."
dl "$ASR/whisper-medium_audio_encode_duration_30s_sr_16k_w8_nash-p_corenum_4.hbm" \
   "$OSS/1.0.0/models/whisper-medium/w8/whisper-medium_audio_encode_duration_30s_sr_16k_w8_nash-p_corenum_4.hbm"
dl "$ASR/whisper-medium_audio_decode_w8_nash-p_corenum_1_1.hbm" \
   "$OSS/1.0.0/models/whisper-medium/w8/whisper-medium_audio_decode_w8_nash-p_corenum_1_1.hbm"

# ---- TTS: WeTTS (CPU onnx) ----
TTS="$MODELS_DIR/wetts_tts"; mkdir -p "$TTS/lib"
echo "[*] WeTTS (TTS) ..."
dl "$TTS/lib/libtts.so"                "$HOBOT_TTS/libtts.so"
dl "$TTS/lib/libonnxruntime.so.1.11.1" "$HOBOT_TTS/libonnxruntime.so.1.11.1"
ln -sf libonnxruntime.so.1.11.1 "$TTS/lib/libonnxruntime.so.1"
ln -sf libonnxruntime.so.1.11.1 "$TTS/lib/libonnxruntime.so"
if [ ! -d "$TTS/tts_model" ]; then
  dl "$TTS/tts_model.tar.gz" "$WETTS_MODEL_URL"; tar xf "$TTS/tts_model.tar.gz" -C "$TTS" && rm -f "$TTS/tts_model.tar.gz"
fi

# ---- 检测: yolo26x_demo (BPU hbm + 官方 runtime) ----
echo "[*] yolo26x_demo (检测) ..."
if [ ! -d "$MODELS_DIR/yolo26x_demo" ]; then
  dl "$MODELS_DIR/yolo26x_demo.tar" "$YOLO26X_DEMO_URL"
  tar xf "$MODELS_DIR/yolo26x_demo.tar" -C "$MODELS_DIR" && rm -f "$MODELS_DIR/yolo26x_demo.tar"
fi

echo "[✓] 完成。"
du -sh "$VLM" "$LLM" "$ASR" "$TTS" "$MODELS_DIR/yolo26x_demo" 2>/dev/null || true
