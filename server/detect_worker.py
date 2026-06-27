"""把地瓜 yolo26x_demo 的 BPU YOLO26 检测包成常驻检测器（人体跟踪用）。

S600 没有 nano 档的 BPU 预编译模型，这里用官方 yolo26x（nash-p hbm）。检测运行时
（hbm_runtime + cv2 + numpy + demo 的 yolo26_det / utils）都在系统 python3.12，通过把
系统 dist-packages 与 demo 路径追加进 sys.path，在本服务进程内直接 import 调用。

懒加载：首次调用 /v1/detect 时才把模型载上 BPU（占 ~8.4ms/帧前向），没用到就不占 BPU。
person = COCO class 0；与 vlm 子进程里的 Qwen3-VL-8B 实测可同驻 BPU。
"""

import sys
import threading

from . import config


class DetectWorker:
    def __init__(self):
        self._model = None
        self._np = None
        self._cv2 = None
        self._lock = threading.Lock()

    def _ensure_started(self):
        if self._model is not None:
            return
        # demo 路径唯一，系统包追加到末尾，避免覆盖 uv venv 里已有的同名包
        for p in (str(config.YOLO_RUNTIME_PY), str(config.YOLO_DEMO_ROOT), config.SYSTEM_SITE_PACKAGES):
            if p not in sys.path:
                sys.path.append(p)
        import cv2
        import numpy as np
        from yolo26_det import YOLO26Config, YOLO26Detect

        self._np = np
        self._cv2 = cv2
        self._model = YOLO26Detect(YOLO26Config(model_path=str(config.YOLO_HBM)))

    def detect(self, image_bytes: bytes, score_thres: float = 0.35, person_only: bool = True) -> dict:
        with self._lock:
            self._ensure_started()
            arr = self._np.frombuffer(image_bytes, dtype=self._np.uint8)
            img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)  # BGR
            if img is None:
                raise ValueError("无法解码图片")
            height, width = img.shape[:2]
            xyxy, score, cls = self._model.predict(img, image_format="BGR", score_thres=score_thres)
        detections = []
        for box, sc, c in zip(xyxy.tolist(), score.tolist(), cls.tolist()):
            class_id = int(c)
            if person_only and class_id != config.PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append({
                "box": [x1, y1, x2, y2],
                "score": float(sc),
                "class_id": class_id,
                "class": "person" if class_id == config.PERSON_CLASS_ID else str(class_id),
            })
        return {"width": width, "height": height, "detections": detections}

    def stop(self):
        self._model = None
