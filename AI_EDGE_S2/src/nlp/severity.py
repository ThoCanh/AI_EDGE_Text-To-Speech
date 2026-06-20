"""
Severity Detector - Phát hiện mức độ nghiêm trọng từ nội dung text.

Dùng để điều chỉnh prosody TTS:
    - critical: nói nhanh + pitch ổn định (khẩn cấp)
    - warning: nhanh hơn bình thường
    - normal: bình thường
"""

from __future__ import annotations


class SeverityDetector:
    """
    Phân loại severity từ text dựa trên keyword matching.

    Example::

        severity = SeverityDetector.detect("Lỗi critical: hệ thống nguy hiểm")
        # → "critical"
    """

    CRITICAL_KEYWORDS = frozenset([
        "khẩn cấp", "critical", "nguy hiểm", "emergency",
        "overheat", "cháy", "ngắn mạch", "short circuit",
    ])

    WARNING_KEYWORDS = frozenset([
        "cảnh báo", "warning", "lỗi", "error", "fault",
        "timeout", "mất kết nối", "không phản hồi",
    ])

    @classmethod
    def detect(cls, text: str) -> str:
        """
        Phát hiện severity level từ text.

        Returns:
            "critical", "warning", hoặc "normal".
        """
        text_lower = text.lower()
        if any(kw in text_lower for kw in cls.CRITICAL_KEYWORDS):
            return "critical"
        if any(kw in text_lower for kw in cls.WARNING_KEYWORDS):
            return "warning"
        return "normal"
