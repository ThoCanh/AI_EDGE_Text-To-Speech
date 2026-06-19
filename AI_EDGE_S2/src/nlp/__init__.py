"""
NLP module - Text Normalization & Code-switching.

Exports:
    CodeSwitchNormalizer: Chuyển đổi text Anh → phiên âm Việt (Khối 3).
    SeverityDetector: Phát hiện mức độ nghiêm trọng từ text.
"""

from .normalizer import CodeSwitchNormalizer
from .severity import SeverityDetector

__all__ = ["CodeSwitchNormalizer", "SeverityDetector"]
