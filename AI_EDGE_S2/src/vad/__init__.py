"""
VAD module - Voice Activity Detection.

Exports:
    SileroVADEngine: Silero VAD wrapper (ONNX).
    VADState: Enum trạng thái state machine.
    VADResult: Kết quả xử lý 1 chunk.
"""

from .types import VADState, VADResult

# Lazy import cho SileroVADEngine (cần ONNX Runtime)
try:
    from .silero_vad import SileroVADEngine
except ImportError:
    SileroVADEngine = None  # type: ignore

__all__ = ["SileroVADEngine", "VADState", "VADResult"]
