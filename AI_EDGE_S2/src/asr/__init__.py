"""
ASR module - Automatic Speech Recognition.

Exports:
    SenseVoiceASR: SenseVoiceSmall wrapper (sherpa-onnx).
"""

from .sensevoice import SenseVoiceASR

__all__ = ["SenseVoiceASR"]
