"""
asr_engine.py — ASR Engine Wrapper (whisper.cpp + Q5_0 GGUF)
=============================================================
Khối 1 + 2 bài test: Model Quantization + Inference Engine

Nhiệm vụ:
- Wrap whisper.cpp thông qua ctypes FFI (Foreign Function Interface)
- Load model Q5_0 GGUF vào RAM 1 lần duy nhất
- Nhận raw PCM float32 data trực tiếp (không cần file .wav)
- Tận dụng ARM NEON SIMD qua whisper.cpp compiled binary

═══════════════════════════════════════════════════════════════
DEEP DIVE: Tại sao Q5_0 cho Whisper-Tiny trên Pi5?
═══════════════════════════════════════════════════════════════

Cortex-A76 trên Pi5 có kiến trúc cache:
  - L1 Data Cache:  64KB per core (4-way set associative)
  - L2 Cache:       512KB per core
  - L3 Cache:       2MB shared across 4 cores

Whisper-Tiny model sizes sau quantization:
  - FP32:   ~150MB  →  không fit L3, constant RAM fetches
  - FP16:   ~75MB   →  KHÔNG NÊN: A76 không có FP16 ALU
  - Q8_0:   ~42MB   →  vẫn lớn, nhiều cache miss
  - Q5_0:   ~30MB   →  ✅ fit tốt hơn trong working set
  - Q4_0:   ~25MB   →  ⚠️ WER degradation quá lớn cho tiếng Việt

Tại sao Q5 nhanh hơn Q8 dù computation phức tạp hơn:
  → Bottleneck trên ARM CPU là MEMORY BANDWIDTH, không phải compute
  → Model nhỏ hơn = ít bytes cần fetch từ RAM = ít cache miss
  → Thời gian fetch từ RAM (~100ns) >> thời gian dequantize (~1ns)
  → Net effect: Q5 nhanh hơn Q8 trên memory-bound workload

Tại sao không Q4:
  → Whisper-Tiny chỉ có 39M parameters (rất nhỏ)
  → Mỗi parameter mang nhiều thông tin hơn so với model lớn
  → Nén xuống 4-bit gây information loss không tỉ lệ
  → Tiếng Việt là low-resource language trong Whisper training data
  → WER delta Q4 có thể đạt 2.5% — vượt ngưỡng yêu cầu 2%
═══════════════════════════════════════════════════════════════
"""

import ctypes
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional

from config import (
    NUM_INFERENCE_THREADS,
    AUDIO_SAMPLE_RATE,
    ASR_MODEL_PATH,
)

logger = logging.getLogger(__name__)


class ASREngine:
    """
    Automatic Speech Recognition engine sử dụng whisper.cpp.
    
    Model được load 1 LẦN DUY NHẤT trong __init__() và giữ trong RAM
    suốt lifetime của object. Các lần inference sau chỉ truyền PCM data
    qua pointer — KHÔNG đọc model từ disk.
    
    ═══════════════════════════════════════════════════════════
    DEEP DIVE: num_threads = 2 trên Pi5
    ═══════════════════════════════════════════════════════════
    
    Raspberry Pi 5 có 4 nhân Cortex-A76. Trực giác nói rằng 
    num_threads=4 sẽ nhanh nhất. NHƯNG KHÔNG:
    
    1) CACHE THRASHING:
       Mỗi core A76 có L2 cache riêng 512KB. Khi 4 threads cùng 
       đọc model weights (30MB Q5_0), mỗi thread cần fetch các 
       phần khác nhau của model vào L2 cache riêng.
       
       4 threads × dữ liệu khác nhau = cache lines bị evict liên tục
       → Mỗi thread phải re-fetch từ RAM → latency tăng
       
       2 threads: cache pressure giảm 50%, hit rate cao hơn
    
    2) MEMORY BANDWIDTH SATURATION:
       Pi5 LPDDR4X bandwidth: ~34 GB/s (shared cho tất cả cores)
       
       AI inference là MEMORY-BOUND workload:
       - Mỗi layer: đọc weights + activations từ RAM
       - Compute chỉ chiếm ~10-20% tổng thời gian
       - 80-90% thời gian = chờ data từ RAM
       
       2 threads đã có thể saturate memory bandwidth
       → Thread 3, 4 chỉ thêm contention, không thêm throughput
    
    3) FORK-JOIN OVERHEAD:
       whisper.cpp dùng OpenMP/pthreads cho parallel GEMM.
       Mỗi matrix multiplication:
         fork → distribute work → compute → join (barrier sync)
       
       Whisper-Tiny có ~150 GEMM ops per forward pass.
       Overhead per barrier: ~1-5μs
       Total overhead: 150 × 5μs = 750μs (4 threads)
       vs:             150 × 2μs = 300μs (2 threads)
       
       Với small model, overhead này ĐÁNG KỂ so với compute time.
    
    4) OS/AUDIO SCHEDULING:
       Pipeline cần CPU cho: audio capture (ALSA), TTS subprocess,
       OS kernel, Python GIL. Nếu dùng hết 4 cores cho ASR:
       → Audio callback bị delay → recording artifacts
       → TTS subprocess bị starve → tổng latency tăng
    
    Kết luận: num_threads = N_cores / 2 = 2 cho small models trên ARM
    ═══════════════════════════════════════════════════════════
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_threads: Optional[int] = None,
        language: str = "vi",
    ):
        """
        Load Whisper-Tiny Q5_0 model vào RAM. GỌI 1 LẦN DUY NHẤT.
        
        Args:
            model_path:  Đường dẫn tới file .bin GGUF (Q5_0)
            num_threads: Số threads cho inference (mặc định 2)
            language:    Ngôn ngữ (mặc định "vi" cho tiếng Việt)
        """
        self._model_path = model_path or ASR_MODEL_PATH
        self._num_threads = num_threads or NUM_INFERENCE_THREADS
        self._language = language
        self._ctx = None  # whisper.cpp context pointer
        self._lib = None  # whisper.cpp shared library

        # ──── LOAD MODEL VÀO RAM (1 LẦN DUY NHẤT) ────
        logger.info(
            f"[ASR] ⏳ Loading Whisper-Tiny Q5_0: {self._model_path} "
            f"(threads={self._num_threads})"
        )
        load_start = time.perf_counter()

        self._load_whisper_cpp()

        load_time = time.perf_counter() - load_start
        logger.info(
            f"[ASR] ✅ Model loaded in {load_time:.2f}s "
            f"(kept in RAM, no reload needed)"
        )

    def _load_whisper_cpp(self):
        """
        Load whisper.cpp shared library và model thông qua ctypes.
        
        Cách 1 (Production): ctypes FFI trực tiếp tới libwhisper.so
        Cách 2 (Alternative): pywhispercpp package
        
        Ở đây implement cả 2 để linh hoạt.
        """
        model_file = Path(self._model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"[ASR] Model file không tồn tại: {self._model_path}\n"
                f"Chạy: scripts/quantize_whisper.sh để tạo model Q5_0"
            )

        # ── Cách 1: Thử pywhispercpp (simple API) ──
        try:
            from pywhispercpp.model import Model as WhisperModel

            self._whisper = WhisperModel(
                str(model_file),
                n_threads=self._num_threads,
                language=self._language,
                print_realtime=False,
                print_progress=False,
                # Tối ưu cho embedded:
                no_timestamps=True,        # Bỏ timestamps → nhanh hơn
                single_segment=True,       # 1 segment cho push-to-talk
            )
            self._backend = "pywhispercpp"
            logger.info("[ASR] Backend: pywhispercpp (Python binding)")
            return
        except ImportError:
            logger.debug("[ASR] pywhispercpp không có, thử ctypes...")

        # ── Cách 2: ctypes FFI trực tiếp ──
        try:
            lib_path = self._find_whisper_lib()
            self._lib = ctypes.CDLL(lib_path)
            
            # whisper_init_from_file(path) → context*
            self._lib.whisper_init_from_file.restype = ctypes.c_void_p
            self._lib.whisper_init_from_file.argtypes = [ctypes.c_char_p]
            
            self._ctx = self._lib.whisper_init_from_file(
                str(model_file).encode("utf-8")
            )
            
            if not self._ctx:
                raise RuntimeError("[ASR] whisper_init_from_file trả về NULL")
            
            self._backend = "ctypes"
            logger.info(f"[ASR] Backend: ctypes FFI ({lib_path})")
        except Exception as e:
            raise RuntimeError(
                f"[ASR] Không thể load whisper.cpp: {e}\n"
                f"Build whisper.cpp: cd whisper.cpp && mkdir build && cd build && "
                f"cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4"
            )

    @staticmethod
    def _find_whisper_lib() -> str:
        """Tìm libwhisper shared library."""
        import platform
        
        candidates = []
        if platform.system() == "Linux":
            candidates = [
                "/usr/local/lib/libwhisper.so",
                "./whisper.cpp/build/libwhisper.so",
                "./whisper.cpp/build/src/libwhisper.so",
            ]
        elif platform.system() == "Darwin":
            candidates = [
                "/usr/local/lib/libwhisper.dylib",
                "./whisper.cpp/build/libwhisper.dylib",
            ]
        elif platform.system() == "Windows":
            candidates = [
                "./whisper.cpp/build/Release/whisper.dll",
                "./whisper.cpp/build/bin/Release/whisper.dll",
            ]

        for path in candidates:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError(
            f"[ASR] libwhisper not found. Candidates: {candidates}"
        )

    def transcribe(self, pcm_data: np.ndarray) -> str:
        """
        Chuyển đổi audio PCM → text.
        
        ═══════════════════════════════════════════════════════
        QUAN TRỌNG: Raw PCM data truyền trực tiếp qua RAM
        ═══════════════════════════════════════════════════════
        
        Bài test yêu cầu: "Cách truyền dữ liệu âm thanh thô 
        (Raw PCM data) vào mô hình ASR mà không cần phải ghi 
        thành file temp.wav xuống ổ cứng."
        
        Giải pháp:
        - numpy array float32 được truyền trực tiếp qua C pointer
        - whisper.cpp nhận float* data + int n_samples
        - KHÔNG tạo file .wav trung gian
        - KHÔNG serialize/deserialize qua disk
        - Data flow: microphone → numpy buffer (RAM) → whisper.cpp (RAM)
        
        Args:
            pcm_data: numpy float32 array, 16kHz mono
                      Shape: (n_samples,) 
                      Values: [-1.0, 1.0] normalized
                      
        Returns:
            Transcribed text string
        """
        if pcm_data.size == 0:
            return ""

        # Đảm bảo format đúng: float32, 1D, contiguous
        if pcm_data.dtype != np.float32:
            pcm_data = pcm_data.astype(np.float32)
        if pcm_data.ndim > 1:
            pcm_data = pcm_data.flatten()
        pcm_data = np.ascontiguousarray(pcm_data)

        inference_start = time.perf_counter()

        if self._backend == "pywhispercpp":
            result = self._transcribe_pywhispercpp(pcm_data)
        else:
            result = self._transcribe_ctypes(pcm_data)

        inference_time = time.perf_counter() - inference_start
        audio_duration = len(pcm_data) / AUDIO_SAMPLE_RATE
        rtf = inference_time / audio_duration if audio_duration > 0 else 0

        logger.info(
            f"[ASR] 📝 '{result}' "
            f"(audio={audio_duration:.1f}s, "
            f"inference={inference_time:.3f}s, "
            f"RTF={rtf:.3f})"
        )

        return result

    def _transcribe_pywhispercpp(self, pcm_data: np.ndarray) -> str:
        """Transcribe qua pywhispercpp binding."""
        segments = self._whisper.transcribe(pcm_data)
        # segments là list of Segment objects
        text = " ".join(seg.text for seg in segments).strip()
        return text

    def _transcribe_ctypes(self, pcm_data: np.ndarray) -> str:
        """
        Transcribe qua ctypes FFI — truyền PCM trực tiếp qua pointer.
        
        whisper_full() nhận:
        - ctx:      context pointer (đã load model trong RAM)
        - params:   inference parameters 
        - samples:  float* pointer tới PCM data
        - n_samples: số lượng samples
        
        → Không file I/O, chỉ pointer passing trong RAM.
        """
        # Tạo default params
        self._lib.whisper_full_default_params.restype = ctypes.c_void_p
        # WHISPER_SAMPLING_GREEDY = 0
        params = self._lib.whisper_full_default_params(0)

        # Set số threads
        # whisper_full_params struct offset cho n_threads
        # Đây là simplified — production code nên dùng proper struct binding
        
        # Gọi whisper_full() với PCM data trực tiếp
        self._lib.whisper_full.restype = ctypes.c_int
        self._lib.whisper_full.argtypes = [
            ctypes.c_void_p,    # ctx
            ctypes.c_void_p,    # params  
            ctypes.POINTER(ctypes.c_float),  # samples (float*)
            ctypes.c_int,       # n_samples
        ]

        # numpy array → C float pointer (ZERO COPY — cùng vùng nhớ)
        samples_ptr = pcm_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n_samples = len(pcm_data)

        ret = self._lib.whisper_full(
            self._ctx, params, samples_ptr, n_samples
        )

        if ret != 0:
            logger.error(f"[ASR] whisper_full() failed: {ret}")
            return ""

        # Lấy kết quả text
        self._lib.whisper_full_n_segments.restype = ctypes.c_int
        self._lib.whisper_full_n_segments.argtypes = [ctypes.c_void_p]
        n_segments = self._lib.whisper_full_n_segments(self._ctx)

        self._lib.whisper_full_get_segment_text.restype = ctypes.c_char_p
        self._lib.whisper_full_get_segment_text.argtypes = [
            ctypes.c_void_p, ctypes.c_int
        ]

        segments = []
        for i in range(n_segments):
            text = self._lib.whisper_full_get_segment_text(self._ctx, i)
            if text:
                segments.append(text.decode("utf-8"))

        return " ".join(segments).strip()

    def warmup(self):
        """
        Pre-warm inference pipeline với dummy audio.
        
        Lần inference đầu tiên luôn chậm hơn do:
        1. JIT compilation (nếu dùng ONNX)
        2. Memory page faults (OS chưa map physical pages)
        3. CPU cache cold (chưa có model data trong L1/L2)
        
        Chạy 1 lần dummy inference để "warm" tất cả các layer này.
        """
        logger.info("[ASR] 🔥 Warming up inference engine...")
        dummy = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)  # 1s silence
        self.transcribe(dummy)
        logger.info("[ASR] ✅ Warm-up complete (cache primed)")

    def shutdown(self):
        """Giải phóng model context khỏi RAM."""
        if self._backend == "ctypes" and self._ctx:
            self._lib.whisper_free.argtypes = [ctypes.c_void_p]
            self._lib.whisper_free(self._ctx)
            self._ctx = None
        elif self._backend == "pywhispercpp":
            del self._whisper
        
        logger.info("[ASR] 🔴 ASR engine shutdown")
