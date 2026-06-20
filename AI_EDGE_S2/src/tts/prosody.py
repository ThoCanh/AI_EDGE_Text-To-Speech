"""
Prosody Controller - Điều chỉnh ngữ điệu TTS theo thời gian thực.

Thay đổi prosody qua scalar parameters TRƯỚC inference:
    - length_scale: tốc độ đọc (< 1.0 = nhanh hơn)
    - noise_scale: biến thiên pitch (thấp = ổn định, nghiêm túc)
    - noise_scale_w: biến thiên duration

KHÔNG tăng inference time vì chỉ là scalar multiplication
trong forward pass của VITS2.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..config import TTS


class ProsodyController:
    """
    Điều khiển ngữ điệu TTS theo severity level.

    Example::

        ctrl = ProsodyController()
        params = ctrl.get_params("critical")
        # → {"length_scale": 0.7, "noise_scale": 0.3, "noise_scale_w": 0.4}
    """

    def __init__(self, profiles: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self._profiles = profiles or TTS.prosody_profiles

    def get_params(self, severity: str) -> Dict[str, float]:
        """
        Lấy tham số prosody cho severity level.

        Args:
            severity: "normal", "warning", hoặc "critical".

        Returns:
            Dict chứa length_scale, noise_scale, noise_scale_w.
        """
        return self._profiles.get(severity, self._profiles["normal"])

    @property
    def available_profiles(self) -> list[str]:
        return list(self._profiles.keys())
