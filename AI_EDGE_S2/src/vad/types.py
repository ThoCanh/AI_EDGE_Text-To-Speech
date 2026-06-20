"""
VAD Data Types - Enum và NamedTuple cho VAD state machine.

Tách riêng để:
- Test được mà không cần ONNX Runtime
- Import nhẹ (chỉ cần Python stdlib)
"""

from __future__ import annotations

import enum
from typing import NamedTuple


class VADState(enum.Enum):
    """Trạng thái của VAD state machine."""

    SILENCE = "SILENCE"
    PENDING_SPEECH = "PENDING_SPEECH"
    SPEECH = "SPEECH"
    PENDING_SILENCE = "PENDING_SILENCE"


class VADResult(NamedTuple):
    """Kết quả xử lý 1 chunk audio qua VAD."""

    is_speech: bool
    state: VADState
    probability: float
    is_end_of_utterance: bool   # True khi chuyển PENDING_SILENCE → SILENCE
