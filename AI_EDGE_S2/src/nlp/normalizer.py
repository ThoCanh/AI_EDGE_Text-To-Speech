"""
Code-switching Text Normalizer - Xử lý đa ngôn ngữ Anh-Việt.

Giải quyết Khối 3: TTS model nhỏ (< 100M params) không thể nhúng
toàn bộ từ điển tiếng Anh. Chiến lược Regex/Rules-based:

    1. Chuyển mọi text tiếng Anh → phiên âm Việt TRƯỚC khi vào TTS
    2. TTS chỉ cần xử lý 1 ngôn ngữ duy nhất (tiếng Việt)
    3. CPU cost ≈ 0 (regex trên string ngắn = microseconds)

Pipeline: Input → Units → Acronyms → Tech Terms → Fallback → Output

Performance Notes:
    - Tất cả regex được pre-compile trong __init__() (1 lần duy nhất)
    - _expand_acronyms dùng dict.get() O(1) thay vì loop O(n)
    - _transliterate_tech_terms dùng 1 combined regex thay vì N regex riêng lẻ
"""

from __future__ import annotations

import logging
import re
from typing import Dict

from .dictionaries import ACRONYM_MAP, TECH_TERMS, UNITS, LETTER_MAP, VIETNAMESE_PASSTHROUGH

logger = logging.getLogger(__name__)

# Sentinel dùng để đánh dấu vùng đã transliterate → fallback bỏ qua
_MARKER_START = "\x00<"
_MARKER_END = ">\x00"


class CodeSwitchNormalizer:
    """
    Text Normalizer cho code-switching Anh-Việt trong ngữ cảnh xe điện.

    Toàn bộ regex được pre-compile 1 lần duy nhất trong __init__().
    normalize() chỉ chạy regex.sub() → O(n) trên string, CPU ≈ microseconds.

    Example::

        norm = CodeSwitchNormalizer()
        result = norm.normalize(
            "Phát hiện lỗi Overcurrent trên đường nguồn 24V"
        )
        # → "Phát hiện lỗi ô-vơ-ca-rần trên đường nguồn 24 vôn"
    """

    def __init__(self) -> None:
        # ── Pre-compile: Unit regex ──────────────────
        sorted_units = sorted(UNITS.keys(), key=len, reverse=True)
        unit_pattern = "|".join(re.escape(u) for u in sorted_units)
        self._unit_re = re.compile(rf"(\d+\.?\d*)\s*({unit_pattern})\b")

        # ── Pre-compile: Acronym regex + lookup dict ─
        self._acronym_re = re.compile(r"\b[A-Z][A-Z0-9]{1,4}\b")
        # Normalize keys to uppercase for O(1) lookup
        self._acronym_lookup: Dict[str, str] = {
            k.upper(): v for k, v in ACRONYM_MAP.items()
        }

        # ── Pre-compile: Tech terms (1 combined regex) ──
        # Sort by length descending → "overcurrent" trước "current"
        sorted_terms = sorted(TECH_TERMS.keys(), key=len, reverse=True)
        if sorted_terms:
            term_pattern = "|".join(re.escape(t) for t in sorted_terms)
            self._tech_re = re.compile(
                rf"\b({term_pattern})\b", re.IGNORECASE,
            )
        else:
            self._tech_re = None
        # Normalize keys to lowercase for O(1) lookup
        self._tech_lookup: Dict[str, str] = {
            k.lower(): v for k, v in TECH_TERMS.items()
        }

        # ── Pre-compile: Fallback English word ───────
        self._english_word_re = re.compile(r"\b[A-Za-z]{2,}\b")

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
            5. Cleanup spacing + remove markers
        """
        if not text:
            return ""

        result = text
        result = self._normalize_units(result)
        result = self._expand_acronyms(result)
        result = self._transliterate_tech_terms(result)
        result = self._fallback_remaining_english(result)
        # Remove markers và cleanup spacing
        result = result.replace(_MARKER_START, "").replace(_MARKER_END, "")
        result = re.sub(r"\s+", " ", result).strip()

        logger.debug("Normalized: '%s' → '%s'", text, result)
        return result

    # ─────────────────────────────────────────
    # Pipeline stages
    # ─────────────────────────────────────────

    def _normalize_units(self, text: str) -> str:
        """24V → 24 vôn, 100mA → 100 mi-li am-pe."""
        def _replace(m: re.Match) -> str:
            vn_unit = UNITS.get(m.group(2), m.group(2))
            return f"{m.group(1)} {_MARKER_START}{vn_unit}{_MARKER_END}"
        return self._unit_re.sub(_replace, text)

    def _expand_acronyms(self, text: str) -> str:
        """BMS → bi em ét. O(1) dict lookup per match."""
        def _replace(m: re.Match) -> str:
            word = m.group(0).upper()
            vn = self._acronym_lookup.get(word)
            if vn is not None:
                return f"{_MARKER_START}{vn}{_MARKER_END}"
            return m.group(0)
        return self._acronym_re.sub(_replace, text)

    def _transliterate_tech_terms(self, text: str) -> str:
        """Overcurrent → ô-vơ-ca-rần. Single pre-compiled regex."""
        if self._tech_re is None:
            return text

        def _replace(m: re.Match) -> str:
            vn = self._tech_lookup[m.group(0).lower()]
            return f"{_MARKER_START}{vn}{_MARKER_END}"
        return self._tech_re.sub(_replace, text)

    def _fallback_remaining_english(self, text: str) -> str:
        """Từ tiếng Anh còn lại → spell out. Bỏ qua vùng đã marker."""
        def _spell(m: re.Match) -> str:
            # Kiểm tra vị trí: nếu nằm trong vùng marker → bỏ qua
            start = m.start()
            # Tìm marker gần nhất trước vị trí match
            marker_start = text.rfind(_MARKER_START, 0, start)
            marker_end = text.rfind(_MARKER_END, 0, start)
            if marker_start >= 0 and (marker_end < 0 or marker_start > marker_end):
                # Đang nằm trong vùng đã transliterate → bỏ qua
                return m.group(0)

            word = m.group(0)
            if self._is_vietnamese_word(word):
                return word
            # Kiểm tra Vietnamese passthrough (từ Việt không dấu)
            if word.lower() in VIETNAMESE_PASSTHROUGH:
                return word
            return " ".join(LETTER_MAP.get(c.lower(), c) for c in word)
        return self._english_word_re.sub(_spell, text)

    @staticmethod
    def _is_vietnamese_word(word: str) -> bool:
        """Kiểm tra từ có chứa ký tự tiếng Việt (có dấu)."""
        vn_chars = frozenset(
            "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
            "ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
            "ỳýỷỹỵđ"
        )
        return any(c.lower() in vn_chars for c in word)
