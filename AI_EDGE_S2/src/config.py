"""
Cấu hình hệ thống tập trung cho Always-on Voice Pipeline.

Tất cả hằng số, đường dẫn model, và ngưỡng được quản lý tại đây.
Không scatter magic numbers trong code → dễ tune trên Pi5 thực tế.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ═══════════════════════════════════════════════════════════
# Đường dẫn gốc
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class AudioConfig:
    """Cấu hình thu âm - chuẩn cho Silero VAD + SenseVoice."""

    sample_rate: int = 16000        # 16kHz mono (chuẩn ASR)
    channels: int = 1               # Mono
    dtype: str = "float32"          # PCM format
    chunk_ms: int = 32              # 32ms per chunk (Silero VAD optimal)
    buffer_seconds: int = 3         # Ring buffer giữ tối đa 3 giây

    @property
    def chunk_size(self) -> int:
        """Số samples trong 1 chunk."""
        return int(self.sample_rate * self.chunk_ms / 1000)

    @property
    def max_buffer_chunks(self) -> int:
        """Số chunks tối đa trong ring buffer."""
        return int(self.buffer_seconds * 1000 / self.chunk_ms)


@dataclass(frozen=True)
class VADConfig:
    """Cấu hình Silero VAD."""

    model_path: str = str(MODELS_DIR / "silero_vad.onnx")
    threshold_on: float = 0.5       # Speech onset probability
    threshold_off: float = 0.35     # Speech offset probability
    min_speech_ms: int = 250        # Tối thiểu 250ms mới tính là speech
    max_silence_ms: int = 700       # Sau 700ms silence → end of utterance
    pre_roll_ms: int = 300          # Giữ 300ms audio trước speech (tránh cắt đầu)
    onnx_threads: int = 1           # 1 thread ONNX cho VAD → CPU < 5%


@dataclass(frozen=True)
class ASRConfig:
    """Cấu hình SenseVoiceSmall (sherpa-onnx)."""

    model_path: str = str(MODELS_DIR / "sensevoice-small" / "model.int8.onnx")
    tokens_path: str = str(MODELS_DIR / "sensevoice-small" / "tokens.txt")
    num_threads: int = 2            # 2 threads cho ASR (optimal trên Pi5)
    decoding_method: str = "greedy_search"


@dataclass(frozen=True)
class TTSConfig:
    """Cấu hình Valtec-TTS."""

    model_path: str = str(MODELS_DIR / "valtec-tts")
    output_sample_rate: int = 22050
    # Prosody profiles: length_scale, noise_scale, noise_scale_w
    prosody_profiles: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "normal": {
            "length_scale": 1.0,
            "noise_scale": 0.667,
            "noise_scale_w": 0.8,
        },
        "warning": {
            "length_scale": 0.85,   # Nhanh hơn 15%
            "noise_scale": 0.5,     # Pitch ổn định hơn
            "noise_scale_w": 0.6,
        },
        "critical": {
            "length_scale": 0.7,    # Nhanh hơn 30%
            "noise_scale": 0.3,     # Pitch rất ổn định (khẩn cấp)
            "noise_scale_w": 0.4,
        },
    })


@dataclass(frozen=True)
class PipelineConfig:
    """Cấu hình pipeline Producer-Consumer."""

    queue_maxsize: int = 50             # Bounded queue
    vad_timeout_s: float = 10.0         # Max speech duration trước khi force-cut
    drop_threshold: float = 0.8         # Drop frames khi queue đầy 80%
    consumer_get_timeout: float = 0.5   # Timeout cho Queue.get() → không block cứng


@dataclass(frozen=True)
class CPUConfig:
    """Ngưỡng CPU theo đề bài."""

    background_max_percent: float = 40.0    # Background: Audio + VAD ≤ 40%
    active_max_percent: float = 70.0        # Active: ASR + TTS ≤ 70%
    throttle_sleep_bg: float = 0.01         # 10ms sleep khi vượt background
    throttle_sleep_active: float = 0.005    # 5ms sleep khi vượt active
    monitor_interval: float = 1.0           # Kiểm tra CPU mỗi 1 giây


# ═══════════════════════════════════════════════════════════
# Singleton instances → import trực tiếp
# ═══════════════════════════════════════════════════════════
AUDIO = AudioConfig()
VAD = VADConfig()
ASR = ASRConfig()
TTS = TTSConfig()
PIPELINE = PipelineConfig()
CPU = CPUConfig()
