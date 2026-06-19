"""
Code-switching Text Normalizer - Xử lý đa ngôn ngữ Anh-Việt.

Giải quyết Khối 3: TTS model nhỏ (< 100M params) không thể nhúng
toàn bộ từ điển tiếng Anh. Chiến lược Regex/Rules-based:

    1. Chuyển mọi text tiếng Anh → phiên âm Việt TRƯỚC khi vào TTS
    2. TTS chỉ cần xử lý 1 ngôn ngữ duy nhất (tiếng Việt)
    3. CPU cost ≈ 0 (regex trên string ngắn = microseconds)

Pipeline: Input → Units → Acronyms → Tech Terms → Fallback → Output
"""

from __future__ import annotations

import logging
import re

from .dictionaries import ACRONYM_MAP, TECH_TERMS, UNITS, LETTER_MAP

logger = logging.getLogger(__name__)


class CodeSwitchNormalizer:
    """
    Text Normalizer cho code-switching Anh-Việt trong ngữ cảnh xe điện.

    Example::

        norm = CodeSwitchNormalizer()
        result = norm.normalize(
            "Phát hiện lỗi Overcurrent trên đường nguồn 24V"
        )
        # → "Phát hiện lỗi ô-vơ-ca-rần trên đường nguồn 24 vôn"
    """

    def __init__(self) -> None:
        # Pre-compile regex patterns cho hiệu năng
        sorted_units = sorted(UNITS.keys(), key=len, reverse=True)
        unit_pattern = "|".join(re.escape(u) for u in sorted_units)
        self._unit_re = re.compile(rf"(\d+\.?\d*)\s*({unit_pattern})\b")
        self._english_word_re = re.compile(r"\b[A-Za-z]{2,}\b")
        self._acronym_re = re.compile(r"\b[A-Z][A-Z0-9]{1,4}\b")

        logger.info(
            "CodeSwitchNormalizer: %d acronyms, %d terms, %d units.",
            len(ACRONYM_MAP), len(TECH_TERMS), len(UNITS),
        )

    def normalize(self, text: str) -> str:
        """
        Pipeline normalization đa ngôn ngữ.

        Thứ tự (quan trọng):
            1. Số + đơn vị: 24V → 24 vôn
            2. Viết tắt: BMS → bi em ét
            3. Thuật ngữ Anh: Overcurrent → ô-vơ-ca-rần
            4. Fallback: từ Anh chưa biết → spell out
            5. Cleanup spacing
        """
        result = text
        result = self._normalize_units(result)
        result = self._expand_acronyms(result)
        result = self._transliterate_tech_terms(result)
        result = self._fallback_remaining_english(result)
        result = re.sub(r"\s+", " ", result).strip()

        logger.debug("Normalized: '%s' → '%s'", text, result)
        return result

    # ─────────────────────────────────────────
    # Pipeline stages
    # ─────────────────────────────────────────

    def _normalize_units(self, text: str) -> str:
        """24V → 24 vôn, 100mA → 100 mi-li am-pe."""
        def _replace(m: re.Match) -> str:
            return f"{m.group(1)} {UNITS.get(m.group(2), m.group(2))}"
        return self._unit_re.sub(_replace, text)

    def _expand_acronyms(self, text: str) -> str:
        """BMS → bi em ét (ALL CAPS words)."""
        def _replace(m: re.Match) -> str:
            word = m.group(0)
            for key, val in ACRONYM_MAP.items():
                if word.upper() == key.upper():
                    return val
            return word
        return self._acronym_re.sub(_replace, text)

    def _transliterate_tech_terms(self, text: str) -> str:
        """Overcurrent → ô-vơ-ca-rần (case-insensitive)."""
        for eng, vn in TECH_TERMS.items():
            text = re.compile(rf"\b{re.escape(eng)}\b", re.IGNORECASE).sub(vn, text)
        return text

    def _fallback_remaining_english(self, text: str) -> str:
        """Từ tiếng Anh còn lại → spell out."""
        def _spell(m: re.Match) -> str:
            word = m.group(0)
            if self._is_vietnamese_word(word):
                return word
            return " ".join(LETTER_MAP.get(c.lower(), c) for c in word)
        return self._english_word_re.sub(_spell, text)

    @staticmethod
    def _is_vietnamese_word(word: str) -> bool:
        """Kiểm tra từ có chứa ký tự tiếng Việt (có dấu)."""
        vn_chars = set(
            "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
            "ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
            "ỳýỷỹỵđ"
        )
        return any(c.lower() in vn_chars for c in word)
