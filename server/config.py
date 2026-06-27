"""本地服务用到的固定路径与模型文件名。

约定（板端 sunrise 用户）：
- oellm_runtime 来自解压后的 D-Robotics_LLM_S600_1.0.2_SDK，预编译二进制与运行库都在其中。
- 模型 hbm 放在本项目 models/ 下，按模型名分目录。
- 运行期生成的 config json 写到 .runtime/，把相对路径全部解析成绝对路径，
  这样二进制在任意工作目录都能跑。
"""

from pathlib import Path

HOME = Path.home()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 模型 hbm 集中放在 /mnt/models 作本地源（777，跨用户/服务可读），与代码分家
MODELS_DIR = Path("/mnt/models")
RUNTIME_DIR = PROJECT_ROOT / ".runtime"

OELLM_ROOT = HOME / "D-Robotics_LLM_S600_1.0.2_SDK" / "oellm_runtime"
OELLM_LIB = OELLM_ROOT / "lib"
OELLM_CONFIGS = OELLM_ROOT / "configs"

VLM_BIN = OELLM_ROOT / "examples" / "vlm_demo" / "vlm"
WHISPER_BIN = OELLM_ROOT / "examples" / "whisper_demo" / "whisper"

# 端侧大模型必须设的 L2 cache 切分（地瓜文档要求）
HBM_ENV = {
    "LD_LIBRARY_PATH": str(OELLM_LIB),
    "HB_DNN_USER_DEFINED_L2M_SIZES": "6:6:6:6",
}

# ---- Qwen3-VL-4B-Instruct (w4) ----
# v1（planner 线上版）：planner 让出 BPU 后视觉由 2B 升到 4B（比 2B 大一点）。
# 不上 8B：8B 的 hbm 太大，与 whisper 同驻 12GB carveout 时 whisper encode
# 模型 hbDNNInitializeFromFiles 失败（BPU 装不下）；4B + whisper + yolo 才能共存。
VLM_DIR = MODELS_DIR / "Qwen3-VL-4B-Instruct"
VLM_MODEL_ID = "Qwen3-VL-4B-Instruct"
VLM_CONFIG = {
    "model_type": "Qwen3-VL",
    "model_dir": str(VLM_DIR) + "/",
    "vit_model_file": "Qwen3-VL-4B-Instruct_vision_448x448_w8-4_nash-p_corenum_4.hbm",
    "llm_model_file": "Qwen3-VL-4B-Instruct_language_chunk_512_cache_1024_w4_nash-p_corenum_4_4.hbm",
    "embed_weight_file_path": "Qwen3-VL-4B-Instruct_embed_tokens_w4_fp16.bin",
    "vit_bpu_core": [0, 1, 2, 3],
    "prefill_bpu_core": [0, 1, 2, 3],
    "decode_bpu_core": [0, 1, 2, 3],
    "vocabulary_path": str(OELLM_CONFIGS / "Qwen3_VL_config"),
    "text_end_token": "<|endoftext|>",
    "img_start_token": "<|vision_start|>",
    "img_end_token": "<|vision_end|>",
    "img_context_token": "<|image_pad|>",
    "mask_pad_value": -32768,
    "temporal_patch_size": 2,
    "patch_size": 16,
    "vocab_size": 151936,
    "embed_dim": 2560,
    "image_height": 448,
    "image_width": 448,
    "image_net_mean": [0.5, 0.5, 0.5],
    "image_net_std": [0.5, 0.5, 0.5],
}

# 大脑 planner 不在本服务内：S600 部署 v1 让 planner 走线上 OpenAI 兼容大模型，
# 本服务只承载端侧的视觉/语音/检测，不再加载本地 LLM planner。

# ---- YOLO26 检测（BPU，取自地瓜 yolo26x_demo；S600 无 nano 档 BPU 预编译，用 x） ----
# 人体跟踪等用：person=COCO class 0。运行时 hbm_runtime/cv2/numpy 与 demo 的 yolo26_det/utils
# 都在系统 python3.12（dist-packages），把这些路径加进 sys.path 即可在本服务进程内 import。
SYSTEM_SITE_PACKAGES = "/usr/local/lib/python3.12/dist-packages"
YOLO_DEMO_ROOT = MODELS_DIR / "yolo26x_demo"
YOLO_HBM = YOLO_DEMO_ROOT / "ultralytics_yolo26" / "model" / "yolo26x_nashp_640x640_nv12.hbm"
YOLO_RUNTIME_PY = YOLO_DEMO_ROOT / "ultralytics_yolo26" / "runtime" / "python"
DETECT_MODEL_ID = "yolo26x"
PERSON_CLASS_ID = 0  # COCO person

# ---- WeTTS (CPU onnx, 取自 hobot_tts；S600 无 BPU TTS 模型，TTS 走 CPU) ----
WETTS_DIR = MODELS_DIR / "wetts_tts"
WETTS_LIB = WETTS_DIR / "lib" / "libtts.so"
WETTS_ONNXRUNTIME = WETTS_DIR / "lib" / "libonnxruntime.so.1.11.1"
WETTS_MODEL_TOP = WETTS_DIR / "tts_model"
WETTS_FLAGS = "tts.flags"  # 相对 WETTS_MODEL_TOP
TTS_MODEL_ID = "wetts-vits-zh"

# ---- whisper-medium (w8) ----
WHISPER_DIR = MODELS_DIR / "whisper-medium"
WHISPER_MODEL_ID = "whisper-medium"
WHISPER_CONFIG = {
    "model_dir": str(WHISPER_DIR) + "/",
    "encode_model_file": "whisper-medium_audio_encode_duration_30s_sr_16k_w8_nash-p_corenum_4.hbm",
    "decode_model_file": "whisper-medium_audio_decode_w8_nash-p_corenum_1_1.hbm",
    "encode_bpu_core": [0, 1, 2, 3],
    "decode_bpu_core": [0],
    "vocabulary_path": str(OELLM_CONFIGS / "Whisper_Medium_config") + "/",
    "language": "zh",
}
