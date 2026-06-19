# -*- coding: utf-8 -*-
"""
demo_windows.py - Demo chay tren Windows (test truoc khi deploy Pi5)
=====================================================================
Luong: Push-to-Talk -> ASR (Whisper-Tiny) -> TTS -> Speaker

Tren Windows:
  - ASR: pywhispercpp (whisper.cpp pre-built DLL)
  - TTS: pyttsx3 (Windows SAPI5) thay Piper (Linux only)
  - Audio: sounddevice (PortAudio)

Tren Pi5 (production):
  - ASR: whisper.cpp native build + ARM NEON
  - TTS: Piper TTS (Vietnamese ONNX)
  - Audio: ALSA
"""

import sys
import os
import gc
import time
import threading
import numpy as np

# Force unbuffered stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import AUDIO_SAMPLE_RATE, NUM_INFERENCE_THREADS, TARGET_RTF

# ============================================================
# CONSTANTS
# ============================================================
SAMPLE_RATE = AUDIO_SAMPLE_RATE  # 16000 Hz (Whisper requirement)
CHANNELS = 1
BLOCK_SIZE = 1024


# ============================================================
# CLASS: VoicePipeline (demo version for Windows)
# ============================================================
class VoicePipeline:
    """
    Voice-to-Voice Pipeline (Push-to-Talk).

    Thiet ke theo yeu cau bai test MET:
      1. Model load 1 lan duy nhat trong __init__()
      2. start_recording() / stop_and_process() cho Push-to-Talk
      3. Raw PCM truyen truc tiep trong RAM (khong file temp.wav)
    """

    def __init__(self):
        """
        ============================================
        WARM-UP: Load tat ca models vao RAM (1 LAN)
        ============================================
        """
        boot_start = time.perf_counter()

        print("=" * 55)
        print("  VOICE-TO-VOICE PIPELINE - WARM UP")
        print(f"  Threads: {NUM_INFERENCE_THREADS}")
        print(f"  ASR: Whisper-Tiny (pywhispercpp)")
        print(f"  TTS: pyttsx3 (Windows demo)")
        print("=" * 55)

        # ---- 1. Load ASR Model (1 LAN DUY NHAT) ----
        print("[ASR] Loading Whisper-Tiny...")
        from pywhispercpp.model import Model as WhisperModel

        self._asr = WhisperModel(
            "tiny",
            n_threads=NUM_INFERENCE_THREADS,  # 2 threads
            language="vi",
            print_realtime=False,
            print_progress=False,
            no_timestamps=True,
            single_segment=True,
        )
        print("[ASR] Model loaded (kept in RAM)")

        # ---- 2. Load TTS Engine (1 LAN DUY NHAT) ----
        print("[TTS] Loading TTS engine...")
        try:
            import pyttsx3
            self._tts = pyttsx3.init()
            self._tts.setProperty("rate", 170)
            self._tts_backend = "pyttsx3"
            print("[TTS] Backend: pyttsx3 (Windows SAPI5)")
        except Exception:
            self._tts = None
            self._tts_backend = "none"
            print("[TTS] No TTS available - text only mode")

        # ---- 3. Audio I/O ----
        import sounddevice as sd
        self._sd = sd
        self._buffer = bytearray()
        self._is_recording = False
        self._lock = threading.Lock()
        self._stream = None

        # ---- 4. Warm-up ASR (prime CPU cache) ----
        print("[ASR] Warming up inference...")
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self._asr.transcribe(dummy)
        print("[ASR] Warm-up complete")

        # ---- 5. Stats ----
        self._cycle_count = 0
        self._baseline_rss = self._get_rss_mb()

        boot_time = time.perf_counter() - boot_start
        print("=" * 55)
        print(f"  READY in {boot_time:.2f}s")
        print(f"  RSS: {self._baseline_rss:.0f}MB")
        print("=" * 55)

    # ============================================
    # PUSH-TO-TALK EVENT HANDLERS
    # ============================================

    def start_recording(self):
        """
        Su kien: Nguoi dung BAM nut (Push).
        Bat dau ghi am PCM vao RAM buffer.
        """
        with self._lock:
            self._buffer = bytearray()
            self._is_recording = True

        self._stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()
        print("\n[REC] Recording... (nhan ENTER de dung)")

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback: Ghi PCM truc tiep vao RAM buffer.
        KHONG ghi file, KHONG temp.wav.

        indata: numpy float32, shape (blocksize, channels)
        -> tobytes() -> bytearray.extend() -> all in RAM
        """
        if self._is_recording:
            self._buffer.extend(indata.tobytes())

    def stop_and_process(self):
        """
        Su kien: Nguoi dung NHA nut (Release).
        Dung ghi -> ASR -> TTS -> Phat loa.
        """
        self._cycle_count += 1
        cycle_start = time.perf_counter()

        # Step 1: Dung recording, lay PCM tu RAM
        with self._lock:
            self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Step 2: Convert buffer -> numpy (trong RAM, khong file)
        if len(self._buffer) == 0:
            print("[PIPELINE] Khong co audio data")
            return

        pcm_data = np.frombuffer(bytes(self._buffer), dtype=np.float32)
        audio_duration = len(pcm_data) / SAMPLE_RATE
        print(f"[REC] Stopped. Duration: {audio_duration:.1f}s")

        # Step 3: ASR - PCM -> Text (model da trong RAM)
        #   pcm_data truyen truc tiep qua RAM pointer
        #   KHONG tao file .wav trung gian
        asr_start = time.perf_counter()
        segments = self._asr.transcribe(pcm_data)
        text = " ".join(seg.text for seg in segments).strip()
        asr_time = time.perf_counter() - asr_start

        print(f"[ASR] Text: '{text}' ({asr_time:.3f}s)")

        if not text:
            print("[PIPELINE] ASR output rong")
            self._cleanup(pcm_data)
            return

        # Step 4: TTS - Text -> Audio -> Speaker
        tts_start = time.perf_counter()
        if self._tts and self._tts_backend == "pyttsx3":
            self._tts.say(text)
            self._tts.runAndWait()
        tts_time = time.perf_counter() - tts_start

        # Step 5: Metrics
        total_inference = asr_time + tts_time
        rtf = total_inference / audio_duration if audio_duration > 0 else 0

        print("-" * 45)
        print(f"  CYCLE #{self._cycle_count} METRICS:")
        print(f"  Input:     {audio_duration:.1f}s")
        print(f"  ASR:       {asr_time:.3f}s")
        print(f"  TTS:       {tts_time:.3f}s")
        rtf_status = "PASS" if rtf < TARGET_RTF else "FAIL"
        print(f"  RTF:       {rtf:.3f} [{rtf_status}] (target < {TARGET_RTF})")
        current_rss = self._get_rss_mb()
        delta = current_rss - self._baseline_rss
        print(f"  Memory:    {current_rss:.0f}MB (delta: {delta:+.1f}MB)")
        print("-" * 45)

        # Step 6: Cleanup
        self._cleanup(pcm_data)

    def _cleanup(self, *arrays):
        """Giai phong buffers trung gian -> tranh memory leak."""
        for arr in arrays:
            del arr
        self._buffer = bytearray()
        gc.collect()

    def _get_rss_mb(self):
        """Lay RSS hien tai (MB)."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def shutdown(self):
        """Giai phong resources."""
        if self._tts and self._tts_backend == "pyttsx3":
            self._tts.stop()
        del self._asr
        gc.collect()
        print("\n[SHUTDOWN] Pipeline da tat.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    pipeline = VoicePipeline()

    print("\n" + "=" * 55)
    print("  PUSH-TO-TALK READY")
    print("  Nhan ENTER de bat dau ghi am")
    print("  Nhan ENTER lan nua de dung va xu ly")
    print("  Ctrl+C de thoat")
    print("=" * 55 + "\n")

    try:
        while True:
            input(">>> [PUSH] Nhan ENTER de ghi am... ")
            pipeline.start_recording()

            input(">>> [RELEASE] Nhan ENTER de xu ly... ")
            pipeline.stop_and_process()

    except KeyboardInterrupt:
        pipeline.shutdown()
    except EOFError:
        pipeline.shutdown()
