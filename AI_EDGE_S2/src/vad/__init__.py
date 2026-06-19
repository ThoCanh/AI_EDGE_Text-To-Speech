"""
VAD module - Voice Activity Detection.

Exports:
    SileroVADEngine: Silero VAD wrapper (ONNX).
    VADState: Enum trạng thái state machine.
    VADResult: Kết quả xử lý 1 chunk.
"""

from .silero_vad import SileroVADEngine, VADState, VADResult

__all__ = ["SileroVADEngine", "VADState", "VADResult"]
