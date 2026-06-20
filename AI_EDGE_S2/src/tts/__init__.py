"""
TTS module - Text-to-Speech với Prosody Control.

Exports:
    ValtecTTSEngine: Valtec-TTS wrapper (Khối 3).
    ProsodyController: Điều chỉnh ngữ điệu theo severity.
"""

from .prosody import ProsodyController
from .valtec_tts import ValtecTTSEngine

__all__ = ["ValtecTTSEngine", "ProsodyController"]
