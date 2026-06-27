"""把 WeTTS（hobot_tts 的 TTS 引擎）的 libtts.so 用 ctypes 常驻起来。

S600 没有 BPU 版 TTS，WeTTS 是 VITS onnx、纯 CPU 推理，跨架构能在 aarch64 跑。
模型与库放在 /mnt/models/wetts_tts（lib/libtts.so + libonnxruntime + tts_model/）。
进程内 init 一次、常驻；合成时加锁串行（VITS 很快，单实例足够）。
"""

import ctypes
import struct
import threading
import wave
from io import BytesIO

from . import config


class _AudioInfo(ctypes.Structure):
    _fields_ = [
        ("sample_rate", ctypes.c_int),
        ("bit_depth", ctypes.c_int),
        ("num_channels", ctypes.c_int),
        ("max_dur_ms", ctypes.c_int),
        ("max_len", ctypes.c_int),
    ]


class TtsWorker:
    def __init__(self):
        self._lib = None
        self._tts = None
        self._info: _AudioInfo | None = None
        self._lock = threading.Lock()

    def start(self):
        # 先把 onnxruntime 以 GLOBAL 方式加载，libtts.so 才能解析到符号
        ctypes.CDLL(str(config.WETTS_ONNXRUNTIME), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(str(config.WETTS_LIB))
        lib.wetts_init.restype = ctypes.c_void_p
        lib.wetts_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        lib.wetts_audio_info.restype = _AudioInfo
        lib.wetts_audio_info.argtypes = [ctypes.c_void_p]
        lib.wetts_synthesis.restype = ctypes.c_int
        lib.wetts_synthesis.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
        ]
        lib.wetts_free.argtypes = [ctypes.c_void_p]

        err = ctypes.c_int(0)
        tts = lib.wetts_init(
            str(config.WETTS_MODEL_TOP).encode(), config.WETTS_FLAGS.encode(), ctypes.byref(err)
        )
        if not tts or err.value != 0x10000:
            raise RuntimeError(f"wetts_init 失败 errcode={hex(err.value)}")
        self._lib = lib
        self._tts = tts
        self._info = lib.wetts_audio_info(tts)

    @property
    def sample_rate(self) -> int:
        return self._info.sample_rate if self._info else 16000

    def synth_wav(self, text: str, speaker_id: int = 0) -> bytes:
        if self._tts is None:
            raise RuntimeError("tts 未就绪")
        with self._lock:
            buf = ctypes.create_string_buffer(self._info.max_len)
            n = ctypes.c_int(0)
            rc = self._lib.wetts_synthesis(
                self._tts, text.encode("utf-8"), speaker_id, buf, ctypes.byref(n)
            )
            if rc != 0x10000:
                raise RuntimeError(f"wetts_synthesis 失败 rc={hex(rc)}")
            nf = n.value
            floats = struct.unpack("<%df" % nf, buf.raw[: nf * 4])
        pcm16 = bytearray()
        for x in floats:
            v = int(max(-1.0, min(1.0, x)) * 32767)
            pcm16 += struct.pack("<h", v)
        out = BytesIO()
        w = wave.open(out, "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(self.sample_rate)
        w.writeframes(bytes(pcm16))
        w.close()
        return out.getvalue()

    def stop(self):
        if self._lib and self._tts:
            self._lib.wetts_free(self._tts)
            self._tts = None
