"""
tts_engine.py — TTS Engine Wrapper (Piper TTS - Vietnamese)
============================================================
Khối 2 bài test: Inference Engine Optimization

Piper TTS chạy như persistent subprocess:
- Model load 1 lần khi khởi tạo subprocess
- Giao tiếp qua stdin/stdout pipe (RAM, không disk I/O)
- Output raw PCM → /dev/shm/ nếu cần file, hoặc trực tiếp tới loa
- Binary đã pre-compiled với ARM NEON optimization
"""

import subprocess
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional

from config import (
    TTS_MODEL_PATH,
    TTS_MODEL_CONFIG,
    PIPER_BINARY,
    TTS_SAMPLE_RATE,
    TTS_LENGTH_SCALE,
    TTS_NOISE_SCALE,
    TTS_NOISE_W,
    TMPFS_DIR,
)
from memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-Speech engine sử dụng Piper TTS.
    
    Thiết kế:
    - Piper chạy như 1 long-lived subprocess (KHÔNG khởi tạo lại mỗi lần)
    - Model ONNX được load vào RAM của subprocess 1 lần duy nhất
    - Input text → stdin pipe → Piper → stdout pipe → raw PCM bytes
    - Toàn bộ data flow qua Unix pipe = RAM buffer của kernel
    
    /dev/shm/ Integration:
    - Khi other processes cần đọc TTS output → ghi vào /dev/shm/
    - Tốc độ R/W: tốc độ RAM (~GB/s), không chạm SD card
    - Tự động cleanup file sau khi đọc xong
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        piper_binary: Optional[str] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """
        Khởi tạo Piper TTS subprocess và load model 1 lần.
        
        Args:
            model_path: Đường dẫn tới ONNX model (.onnx)
            piper_binary: Đường dẫn tới Piper binary
            memory_manager: MemoryManager instance cho /dev/shm/
        """
        self._model_path = model_path or TTS_MODEL_PATH
        self._piper_binary = piper_binary or PIPER_BINARY
        self._mem = memory_manager
        self._process: Optional[subprocess.Popen] = None

        # Validate model file
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"[TTS] Model không tồn tại: {self._model_path}\n"
                f"Download: wget https://huggingface.co/rhasspy/piper-voices/"
                f"resolve/main/vi/vi_VN-vais1000-medium.onnx"
            )

        # ──── KHỞI TẠO SUBPROCESS (LOAD MODEL 1 LẦN) ────
        logger.info(f"[TTS] ⏳ Khởi tạo Piper TTS: {self._model_path}")
        load_start = time.perf_counter()

        self._start_piper_process()

        load_time = time.perf_counter() - load_start
        logger.info(
            f"[TTS] ✅ Piper TTS ready ({load_time:.2f}s). "
            f"Model kept in subprocess RAM."
        )

    def _start_piper_process(self):
        """
        Khởi tạo Piper subprocess dạng long-running.
        
        Piper hỗ trợ chế độ "streaming": đọc text từ stdin,
        output raw PCM ra stdout. Model chỉ load 1 lần khi 
        subprocess start → các request sau chỉ inference.
        
        ──── Output Format ────
        --output-raw: PCM int16, mono, 22050Hz
        → Không cần decode WAV header → tiết kiệm compute
        → Truyền trực tiếp vào sounddevice hoặc ALSA
        """
        cmd = [
            self._piper_binary,
            "--model", self._model_path,
            "--output-raw",                    # Raw PCM output (không WAV)
            "--length-scale", str(TTS_LENGTH_SCALE),    # Tốc độ đọc
            "--noise-scale", str(TTS_NOISE_SCALE),      # Biến thiên giọng
            "--noise-w", str(TTS_NOISE_W),              # Biến thiên duration
        ]

        # Thêm config file nếu có
        if Path(TTS_MODEL_CONFIG).exists():
            cmd.extend(["--config", TTS_MODEL_CONFIG])

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered để giảm latency
            )
            logger.info(f"[TTS] Piper subprocess PID: {self._process.pid}")
        except FileNotFoundError:
            raise RuntimeError(
                f"[TTS] Piper binary không tìm thấy: {self._piper_binary}\n"
                f"Download pre-built: wget https://github.com/rhasspy/piper/"
                f"releases/download/v1.2.0/piper_arm64.tar.gz"
            )

    def synthesize(self, text: str) -> np.ndarray:
        """
        Chuyển text → raw PCM audio trong RAM.
        
        Data flow (tất cả trong RAM):
        1. text → encode UTF-8 → stdin pipe (kernel buffer)
        2. Piper inference (model đã load sẵn trong RAM)
        3. Raw PCM int16 → stdout pipe (kernel buffer) 
        4. Read bytes → numpy array (user-space RAM)
        
        KHÔNG có disk I/O ở bất kỳ bước nào.
        
        Args:
            text: Vietnamese text to synthesize
            
        Returns:
            numpy float32 array, 22050Hz mono, normalized [-1.0, 1.0]
        """
        if not text.strip():
            return np.array([], dtype=np.float32)

        if self._process is None or self._process.poll() is not None:
            logger.warning("[TTS] Subprocess died, restarting...")
            self._start_piper_process()

        inference_start = time.perf_counter()

        # ── Gửi text qua stdin pipe ──
        # Piper đọc 1 line = 1 utterance
        text_bytes = (text.strip() + "\n").encode("utf-8")
        self._process.stdin.write(text_bytes)
        self._process.stdin.flush()

        # ── Đọc raw PCM từ stdout pipe ──
        # Piper output format: int16 little-endian, 22050Hz, mono
        # Vấn đề: stdout.read() sẽ block vô hạn
        # Giải pháp: đọc từng chunk cho đến khi Piper flush xong
        raw_pcm = self._read_piper_output()

        if len(raw_pcm) == 0:
            logger.warning("[TTS] Piper output rỗng")
            return np.array([], dtype=np.float32)

        # ── Convert int16 → float32 normalized ──
        audio_int16 = np.frombuffer(raw_pcm, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        inference_time = time.perf_counter() - inference_start
        audio_duration = len(audio_float32) / TTS_SAMPLE_RATE

        logger.info(
            f"[TTS] 🔊 Synthesized '{text[:30]}...' "
            f"(output={audio_duration:.1f}s, "
            f"inference={inference_time:.3f}s, "
            f"samples={len(audio_float32)})"
        )

        return audio_float32

    def _read_piper_output(self) -> bytes:
        """
        Đọc raw PCM output từ Piper stdout pipe.
        
        Challenge: Piper output có variable length, stdout.read() 
        sẽ block vĩnh viễn vì subprocess vẫn alive.
        
        Giải pháp: Dùng selectors để đọc từng chunk với timeout.
        Piper ghi PCM data rồi flush, ta đọc cho đến khi 
        không còn data mới trong pipe.
        """
        import io

        buffer = io.BytesIO()
        
        try:
            import selectors
            sel = selectors.DefaultSelector()
            sel.register(self._process.stdout, selectors.EVENT_READ)
            
            while True:
                events = sel.select(timeout=0.5)  # 500ms timeout
                if not events:
                    break
                
                done = False
                for key, _ in events:
                    chunk = key.fileobj.read1(4096)
                    if chunk:
                        buffer.write(chunk)
                    else:
                        done = True
                if done:
                    break
            
            sel.unregister(self._process.stdout)
            sel.close()
        except (AttributeError, ImportError, ValueError):
            # Fallback: communicate() (subprocess sẽ exit sau 1 request)
            stdout, _ = self._process.communicate()
            buffer.write(stdout)

        return buffer.getvalue()

    def synthesize_to_shm(self, text: str) -> str:
        """
        Synthesize và lưu vào /dev/shm/ cho other processes đọc.
        
        Use case: Khi TTS output cần được process khác sử dụng
        (ví dụ: ALSA aplay, hoặc audio post-processing subprocess).
        
        File được ghi vào /dev/shm/ (tmpfs):
        - Tốc độ: tốc độ RAM (~4-8 GB/s)
        - Không hao mòn MicroSD
        - Tự động mất khi reboot
        
        Returns:
            Path tới file PCM trên /dev/shm/
        """
        audio = self.synthesize(text)
        
        if audio.size == 0:
            return ""

        # Convert back to int16 for raw PCM file
        pcm_int16 = (audio * 32768.0).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        if self._mem:
            path = self._mem.write_temp("tts_output.raw", pcm_bytes)
        else:
            import os
            path = os.path.join(TMPFS_DIR, "voice_pipeline_tts_output.raw")
            with open(path, "wb") as f:
                f.write(pcm_bytes)

        logger.info(
            f"[TTS] 💾 Saved to {path} "
            f"({len(pcm_bytes)} bytes, {len(audio)/TTS_SAMPLE_RATE:.1f}s)"
        )

        return path

    def warmup(self):
        """
        Pre-warm TTS pipeline.
        Gửi 1 câu ngắn để Piper JIT optimize + cache model layers.
        """
        logger.info("[TTS] 🔥 Warming up TTS engine...")
        _ = self.synthesize("Xin chào")
        logger.info("[TTS] ✅ Warm-up complete")

    def shutdown(self):
        """Terminate Piper subprocess và giải phóng resources."""
        if self._process and self._process.poll() is None:
            self._process.stdin.close()
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        
        logger.info("[TTS] 🔴 TTS engine shutdown")
