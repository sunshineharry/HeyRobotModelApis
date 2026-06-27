#!/usr/bin/env bash
# 下载 S600 端侧模型到本地源目录，布局与 server/config.py 一致。
#   - Qwen3-VL-8B-Instruct（VLM，地瓜 BPU hbm，w4）
#   - whisper-medium（ASR，地瓜 BPU hbm，w8）
#   - WeTTS（TTS，CPU onnx，取自 hobot_tts）
# 用法：./download_models.sh            # 默认下到 /mnt/models
#       MODELS_DIR=~/models ./download_models.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/mnt/models}"
OSS="https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s600"
HOBOT_TTS="https://github.com/D-Robotics/hobot_tts/raw/develop/wetts/lib"
WETTS_MODEL_URL="http://archive.d-robotics.cc/tts-model/tts_model.tar.gz"

echo "[*] 模型目录: $MODELS_DIR"
mkdir -p "$MODELS_DIR" 2>/dev/null || sudo mkdir -p "$MODELS_DIR"
[ -w "$MODELS_DIR" ] || sudo chmod 777 "$MODELS_DIR"

dl() { wget -c -q --show-progress -O "$1" "$2"; }

# ---- VLM: Qwen3-VL-8B-Instruct (w4, 1.0.2) ----
VLM="$MODELS_DIR/Qwen3-VL-8B-Instruct"; mkdir -p "$VLM"
echo "[*] Qwen3-VL-8B-Instruct ..."
dl "$VLM/Qwen3-VL-8B-Instruct_vision_448x448_w8-4_nash-p_corenum_4.hbm" \
   "$OSS/1.0.2/models/Qwen3-VL-8B-Instruct/w4/Qwen3-VL-8B-Instruct_vision_448x448_w8-4_nash-p_corenum_4.hbm"
dl "$VLM/Qwen3-VL-8B-Instruct_language_chunk_512_cache_1024_w4_nash-p_corenum_4_4.hbm" \
   "$OSS/1.0.2/models/Qwen3-VL-8B-Instruct/w4/Qwen3-VL-8B-Instruct_language_chunk_512_cache_1024_w4_nash-p_corenum_4_4.hbm"
dl "$VLM/Qwen3-VL-8B-Instruct_embed_tokens_w4_fp16.bin" \
   "$OSS/1.0.2/models/Qwen3-VL-8B-Instruct/w4/Qwen3-VL-8B-Instruct_embed_tokens_w4_fp16.bin"

# ---- ASR: whisper-medium (w8, 1.0.0) ----
ASR="$MODELS_DIR/whisper-medium"; mkdir -p "$ASR"
echo "[*] whisper-medium ..."
dl "$ASR/whisper-medium_audio_encode_duration_30s_sr_16k_w8_nash-p_corenum_4.hbm" \
   "$OSS/1.0.0/models/whisper-medium/w8/whisper-medium_audio_encode_duration_30s_sr_16k_w8_nash-p_corenum_4.hbm"
dl "$ASR/whisper-medium_audio_decode_w8_nash-p_corenum_1_1.hbm" \
   "$OSS/1.0.0/models/whisper-medium/w8/whisper-medium_audio_decode_w8_nash-p_corenum_1_1.hbm"

# ---- TTS: WeTTS (CPU onnx, hobot_tts) ----
TTS="$MODELS_DIR/wetts_tts"; mkdir -p "$TTS/lib"
echo "[*] WeTTS lib + model ..."
dl "$TTS/lib/libtts.so"                 "$HOBOT_TTS/libtts.so"
dl "$TTS/lib/libonnxruntime.so.1.11.1"  "$HOBOT_TTS/libonnxruntime.so.1.11.1"
ln -sf libonnxruntime.so.1.11.1 "$TTS/lib/libonnxruntime.so.1"
ln -sf libonnxruntime.so.1.11.1 "$TTS/lib/libonnxruntime.so"
if [ ! -d "$TTS/tts_model" ]; then
  dl "$TTS/tts_model.tar.gz" "$WETTS_MODEL_URL"
  tar xf "$TTS/tts_model.tar.gz" -C "$TTS" && rm -f "$TTS/tts_model.tar.gz"
fi

echo "[✓] 完成。目录结构："
du -sh "$VLM" "$ASR" "$TTS" 2>/dev/null || true
