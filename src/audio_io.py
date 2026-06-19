"""
audio_io.py — Audio Recording & Playback cho Push-to-Talk
==========================================================
Xử lý audio I/O trên Raspberry Pi 5 qua ALSA/PulseAudio.

Key design:
- Recording: PCM float32 16kHz mono → bytearray buffer trong RAM
- Playback: numpy array → sounddevice/ALSA → speaker
- KHÔNG ghi file .wav trung gian (yêu cầu bài test)
- Callback-based recording để non-blocking
- Tự động phát hiện và chọn đúng microphone
- Lọc tạp âm nhẹ trước khi đưa vào ASR
"""

import os
import numpy as np
import threading
import logging
from typing import Optional

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_BLOCK_SIZE,
    AUDIO_DTYPE,
    TTS_SAMPLE_RATE,
    AUDIO_DEVICE_INDEX,
    AUDIO_DEVICE_NAME,
    NOISE_REDUCE_ENABLED,
)
from noise_filter import NoiseFilter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# MICROPHONE SELECTION
# ═══════════════════════════════════════════════════════════════

def find_microphone(sd_module) -> Optional[int]:
    """
    Tìm và chọn microphone phù hợp nhất.

    Ưu tiên (từ trên xuống):
      1. AUDIO_DEVICE_INDEX (config.py) — chỉ định trực tiếp
      2. AUDIO_DEVICE_NAME (config.py) — tìm theo tên
      3. Auto-detect — ưu tiên USB mic > built-in mic

    Tại sao cần auto-detect:
      - Pi5 thường có 2+ audio devices (HDMI, USB mic, Bluetooth)
      - PC dev có built-in mic, headset, webcam mic...
      - Nếu chọn sai device → ghi âm từ HDMI (không có mic) → ASR rỗng

    Returns:
        device_index (int) hoặc None (dùng default)
    """
    # ── Ưu tiên 1: User chỉ định trực tiếp ──
    if AUDIO_DEVICE_INDEX is not None:
        try:
            info = sd_module.query_devices(AUDIO_DEVICE_INDEX)
            if info["max_input_channels"] > 0:
                logger.info(
                    f"[MIC] Dung device #{AUDIO_DEVICE_INDEX}: "
                    f"'{info['name']}'"
                )
                return AUDIO_DEVICE_INDEX
            else:
                logger.warning(
                    f"[MIC] Device #{AUDIO_DEVICE_INDEX} khong co input!"
                )
        except Exception as e:
            logger.warning(f"[MIC] Device #{AUDIO_DEVICE_INDEX} loi: {e}")

    # ── Ưu tiên 2: Tìm theo tên ──
    if AUDIO_DEVICE_NAME is not None:
        devices = sd_module.query_devices()
        for i, dev in enumerate(devices):
            if (AUDIO_DEVICE_NAME.lower() in dev["name"].lower()
                    and dev["max_input_channels"] > 0):
                logger.info(
                    f"[MIC] Tim thay '{AUDIO_DEVICE_NAME}' -> "
                    f"device #{i}: '{dev['name']}'"
                )
                return i
        logger.warning(
            f"[MIC] Khong tim thay device ten '{AUDIO_DEVICE_NAME}'"
        )

    # ── Ưu tiên 3: Auto-detect (USB mic > built-in) ──
    return _auto_detect_mic(sd_module)


def _auto_detect_mic(sd_module) -> Optional[int]:
    """
    Tự động chọn mic tốt nhất.

    Chiến lược ưu tiên:
      1. USB microphone (phổ biến trên Pi5, chất lượng tốt hơn)
      2. Bất kỳ device nào có input channels
      3. None (để sounddevice chọn default)

    USB mic thường có tên chứa: "USB", "Microphone", "Audio Device"
    """
    devices = sd_module.query_devices()
    usb_candidates = []
    other_candidates = []

    # Virtual devices to skip (not real microphones)
    skip_keywords = ["sound mapper", "primary", "stereo mix", "line in"]

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            name_lower = dev["name"].lower()

            # Skip virtual/system devices
            if any(skip in name_lower for skip in skip_keywords):
                continue

            # USB mic keywords
            if any(kw in name_lower for kw in ["usb", "microphone", "mic"]):
                usb_candidates.append((i, dev))
            else:
                other_candidates.append((i, dev))

    # Chọn USB mic nếu có
    if usb_candidates:
        idx, dev = usb_candidates[0]
        logger.info(
            f"[MIC] Auto-detect USB mic -> device #{idx}: '{dev['name']}'"
        )
        return idx

    # Chọn device input bất kỳ
    if other_candidates:
        idx, dev = other_candidates[0]
        logger.info(
            f"[MIC] Auto-detect input device -> #{idx}: '{dev['name']}'"
        )
        return idx

    # Không tìm thấy → để sounddevice chọn default
    logger.info("[MIC] Dung default audio input device")
    return None


def list_audio_devices():
    """
    In danh sách tất cả audio devices.
    Chạy: python -c "from audio_io import list_audio_devices; list_audio_devices()"
    """
    import sounddevice as sd
    print("\n" + "=" * 60)
    print("  DANH SACH AUDIO DEVICES")
    print("=" * 60)

    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        direction = ""
        if dev["max_input_channels"] > 0:
            direction += "IN "
        if dev["max_output_channels"] > 0:
            direction += "OUT"

        marker = ""
        if i == sd.default.device[0]:
            marker += " [DEFAULT INPUT]"
        if i == sd.default.device[1]:
            marker += " [DEFAULT OUTPUT]"

        print(
            f"  #{i:2d} [{direction:6s}] "
            f"{dev['name']}{marker}"
        )

    print("=" * 60)
    print(f"  Default input:  #{sd.default.device[0]}")
    print(f"  Default output: #{sd.default.device[1]}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# AUDIO RECORDER
# ═══════════════════════════════════════════════════════════════

class AudioRecorder:
    """
    Non-blocking audio recorder sử dụng callback pattern.

    Audio data được ghi trực tiếp vào RAM buffer (bytearray).
    Khi nhả nút, buffer được convert sang numpy array và truyền
    trực tiếp vào ASR engine — KHÔNG qua file trung gian.

    Data flow: Microphone → ALSA → callback → bytearray (RAM)
                          → NoiseFilter → numpy → ASR
    """

    def __init__(self):
        self._buffer = bytearray()
        self._is_recording = False
        self._lock = threading.Lock()
        self._stream = None

        # Import sounddevice lazily (cần ALSA trên Pi5)
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            raise ImportError(
                "[AUDIO] sounddevice chua cai.\n"
                "Pi5: pip install sounddevice\n"
                "Can: sudo apt install libportaudio2"
            )

        # ── Chọn microphone ──
        self._device_index = find_microphone(sd)

        # ── Khởi tạo noise filter ──
        if NOISE_REDUCE_ENABLED:
            self._noise_filter = NoiseFilter()
        else:
            self._noise_filter = None

    def start(self):
        """
        Bắt đầu ghi âm (Push event).

        Mở audio input stream với callback.
        Mỗi lần callback được gọi (~64ms intervals):
        - Nhận 1024 frames PCM float32
        - Append trực tiếp vào bytearray buffer
        - Tất cả trong RAM, không disk I/O
        """
        with self._lock:
            self._buffer = bytearray()  # Reset buffer
            self._is_recording = True

        self._stream = self._sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype=AUDIO_DTYPE,
            blocksize=AUDIO_BLOCK_SIZE,
            device=self._device_index,  # Mic đã chọn
            callback=self._callback,
        )
        self._stream.start()

        dev_name = "default"
        if self._device_index is not None:
            dev_name = f"device #{self._device_index}"
        logger.info(f"[REC] Recording started (16kHz/mono/{dev_name})")

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Audio callback — ghi PCM trực tiếp vào RAM.

        indata: numpy float32 array, shape (blocksize, channels)

        ═══════════════════════════════════════════════════════
        YÊU CẦU BÀI TEST: "Cách truyền dữ liệu âm thanh thô
        (Raw PCM data) vào mô hình ASR mà không cần phải ghi
        thành file temp.wav xuống ổ cứng."

        → indata.tobytes() chuyển numpy array sang bytes
        → bytearray.extend() append vào buffer trong RAM
        → Khi stop, buffer được cast lại thành numpy array
        → Truyền trực tiếp vào whisper.cpp qua C pointer
        → ZERO disk I/O trong toàn bộ recording pipeline
        ═══════════════════════════════════════════════════════
        """
        if status:
            logger.warning(f"[REC] Audio callback status: {status}")

        if self._is_recording:
            self._buffer.extend(indata.tobytes())

    def stop(self) -> np.ndarray:
        """
        Dừng ghi âm (Release event) và trả về PCM data đã lọc tạp âm.

        Returns:
            numpy float32 array (16kHz, mono) — đã lọc noise
            Truyền trực tiếp vào ASR, KHÔNG tạo file trung gian.
        """
        with self._lock:
            self._is_recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # ── Convert bytearray → numpy array (trong RAM) ──
        if len(self._buffer) == 0:
            logger.warning("[REC] Buffer rong — khong ghi duoc am thanh")
            return np.array([], dtype=np.float32)

        pcm_data = np.frombuffer(bytes(self._buffer), dtype=np.float32)

        duration = len(pcm_data) / AUDIO_SAMPLE_RATE

        logger.info(
            f"[REC] Stopped. "
            f"Duration: {duration:.1f}s, "
            f"Samples: {len(pcm_data)}, "
            f"Buffer: {len(self._buffer)} bytes"
        )

        # Clear buffer — giải phóng memory
        self._buffer = bytearray()

        # ── Lọc tạp âm (trong RAM, trước khi đưa vào ASR) ──
        if self._noise_filter is not None:
            pcm_data = self._noise_filter.apply(pcm_data)
            logger.info("[REC] Noise filter applied")

        return pcm_data

    @property
    def is_recording(self) -> bool:
        return self._is_recording


# ═══════════════════════════════════════════════════════════════
# AUDIO PLAYER
# ═══════════════════════════════════════════════════════════════

class AudioPlayer:
    """
    Audio playback qua sounddevice (ALSA backend trên Pi5).

    Hỗ trợ 2 mode:
    1. Direct play: numpy array → sounddevice → speaker
    2. File play:   /dev/shm/file.raw → read → play
    """

    def __init__(self):
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            raise ImportError("[AUDIO] sounddevice chua cai")

    def play(self, audio: np.ndarray, sample_rate: int = TTS_SAMPLE_RATE):
        """
        Phát audio trực tiếp từ numpy array (blocking).

        Args:
            audio: numpy float32 array [-1.0, 1.0]
            sample_rate: Sample rate (22050 cho Piper TTS output)
        """
        if audio.size == 0:
            logger.warning("[PLAY] Audio array rong")
            return

        duration = len(audio) / sample_rate
        logger.info(f"[PLAY] Playing {duration:.1f}s audio @ {sample_rate}Hz")

        self._sd.play(audio, samplerate=sample_rate)
        self._sd.wait()  # Block until playback finishes

        logger.info("[PLAY] Playback complete")

    def play_from_shm(self, shm_path: str, sample_rate: int = TTS_SAMPLE_RATE):
        """
        Phát audio từ file raw PCM trên /dev/shm/.

        Dùng khi TTS output đã lưu vào /dev/shm/ để share
        với other processes. Đọc từ tmpfs = tốc độ RAM.

        Args:
            shm_path: Path tới file trên /dev/shm/
            sample_rate: Sample rate của file
        """
        if not os.path.exists(shm_path):
            logger.error(f"[PLAY] File khong ton tai: {shm_path}")
            return

        with open(shm_path, "rb") as f:
            raw_bytes = f.read()

        audio = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float = audio.astype(np.float32) / 32768.0

        self.play(audio_float, sample_rate)

        # Cleanup file tạm sau khi phát
        try:
            os.remove(shm_path)
            logger.debug(f"[PLAY] Cleaned up: {shm_path}")
        except OSError:
            pass
