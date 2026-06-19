"""
config.py — Hardware-Specific Constants cho Raspberry Pi 5
==========================================================
Tất cả magic numbers và cấu hình phần cứng tập trung tại đây.
Khi deploy lên board khác (Jetson, Orange Pi...) chỉ cần sửa file này.
"""

import os

# ═══════════════════════════════════════════════════════════════
# HARDWARE TARGET: Raspberry Pi 5
# SoC: Broadcom BCM2712
# CPU: 4x ARM Cortex-A76 @ 2.4GHz
# RAM: 4GB / 8GB LPDDR4X
# Cache: 512KB L2 per core (shared), 2MB L3 (shared)
# ISA:  ARMv8.2-A + NEON SIMD + dotprod
# ═══════════════════════════════════════════════════════════════

# ─── CPU / Threading ──────────────────────────────────────────
NUM_PHYSICAL_CORES = 4
NUM_INFERENCE_THREADS = 2  # Tối ưu cho Pi5 (xem docs/deep_dive_num_threads.md)

# Giải thích ngắn:
# - 2 threads cho inference, 2 threads còn lại cho OS/audio/TTS subprocess
# - Tránh cache thrashing khi 4 threads cùng đọc model weights
# - Memory bandwidth LPDDR4X ~34GB/s bão hòa ở 2-3 threads cho AI workload

# ─── Audio Format ─────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000    # Whisper yêu cầu 16kHz
AUDIO_CHANNELS = 1           # Mono
AUDIO_BLOCK_SIZE = 1024      # Frames per buffer callback
AUDIO_DTYPE = "float32"      # PCM format cho whisper.cpp input

# Piper TTS output format
TTS_SAMPLE_RATE = 22050      # Piper default output rate

# ─── Model Paths ──────────────────────────────────────────────
# Relative to project root
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

ASR_MODEL_PATH = os.path.join(MODEL_DIR, "ggml-tiny-q5_0.bin")
TTS_MODEL_PATH = os.path.join(MODEL_DIR, "vi_VN-vais1000-medium.onnx")
TTS_MODEL_CONFIG = os.path.join(MODEL_DIR, "vi_VN-vais1000-medium.onnx.json")

# Piper binary path
PIPER_BINARY = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "piper", "piper"
)

# ─── Quantization ─────────────────────────────────────────────
# Q5_0 GGUF — sweet spot cho Whisper-Tiny trên Pi5
# (xem docs/deep_dive_q5_quantization.md)

# ─── Memory Management ───────────────────────────────────────
# /dev/shm/ — tmpfs (RAM-based filesystem)
# (xem docs/deep_dive_dev_shm.md)
#
# Tại sao dùng /dev/shm/ thay vì /tmp/:
# 1. /dev/shm/ LUÔN là tmpfs (POSIX guarantee)
# 2. /tmp/ có thể mount trên disk tùy distro Linux
# 3. Tốc độ R/W = tốc độ RAM (~GB/s vs SD card ~50MB/s)
# 4. Không ghi xuống storage vật lý → không hao mòn MicroSD
TMPFS_DIR = "/dev/shm"
TTS_TEMP_FILE = os.path.join(TMPFS_DIR, "voice_pipeline_tts_output.raw")

# Fallback cho môi trường dev (Windows/Mac không có /dev/shm/)
if not os.path.exists(TMPFS_DIR):
    import tempfile
    TMPFS_DIR = tempfile.gettempdir()
    TTS_TEMP_FILE = os.path.join(TMPFS_DIR, "voice_pipeline_tts_output.raw")

# Memory monitoring
MEMORY_CHECK_INTERVAL_SECONDS = 30  # Kiểm tra memory leak mỗi 30s
MEMORY_LEAK_THRESHOLD_MB = 10       # Cảnh báo nếu RSS tăng > 10MB

# ─── Performance KPIs ────────────────────────────────────────
TARGET_RTF = 0.3             # Real-Time Factor < 0.3
# RTF = processing_time / audio_duration
# 5s audio → phải xử lý ASR + TTS trong < 1.5s

# ─── TTS Config ──────────────────────────────────────────────
TTS_LENGTH_SCALE = 0.9       # Tốc độ đọc (< 1.0 = nhanh hơn)
TTS_NOISE_SCALE = 0.667      # Độ biến thiên giọng
TTS_NOISE_W = 0.8            # Độ biến thiên phoneme duration

# ─── Microphone Selection ────────────────────────────────
# Trên Pi5/PC thường có nhiều mic: built-in, USB, Bluetooth...
# Cách chọn mic (ưu tiên từ trên xuống):
#   1. AUDIO_DEVICE_INDEX: chỉ định trực tiếp index (0, 1, 2...)
#   2. AUDIO_DEVICE_NAME: tìm theo tên (partial match)
#   3. None: để sounddevice chọn default device
#
# Dùng lệnh sau để xem danh sách thiết bị:
#   python -c "import sounddevice; print(sounddevice.query_devices())"
AUDIO_DEVICE_INDEX = None      # Ví dụ: 1 cho USB mic thứ 2
AUDIO_DEVICE_NAME = None       # Ví dụ: "USB" để tự tìm USB mic

# ─── Noise Reduction ────────────────────────────────────
# Lọc tạp âm/tiếng ồn nhẹ bằng DSP trên CPU (không cần GPU)
# Phù hợp cho ARM Cortex-A76 (Pi5) — tốn rất ít CPU
NOISE_REDUCE_ENABLED = True

# High-pass filter: loại bỏ tần số thấp (tiếng ù, gió, máy lạnh)
# Tần số cắt 80Hz: giữ giọng nói (100-8000Hz), bỏ tiếng ồn nền
HIGHPASS_CUTOFF_HZ = 80

# Noise gate: tắt tiếng khi âm lượng dưới ngưỡng
# Giá trị 0.01 = -40dB, phù hợp phòng yên tĩnh đến vừa
NOISE_GATE_THRESHOLD = 0.01

# Spectral gating: giảm tiếng ồn dựa trên phổ tần
# Aggressive hơn nhưng tốn CPU hơn (tắt nếu RTF > 0.3)
SPECTRAL_GATE_ENABLED = False
