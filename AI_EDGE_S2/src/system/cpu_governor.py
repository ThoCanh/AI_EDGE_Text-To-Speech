"""
CPU Governor - Giám sát và kiểm soát tài nguyên CPU.

Đảm bảo tuân thủ ràng buộc phần cứng:
    - Background (Audio + VAD): ≤ 40% CPU
    - Active (ASR + TTS):       ≤ 70% CPU

Cơ chế: Đo CPU usage theo chu kỳ, adaptive sleep khi vượt ngưỡng.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from ..config import CPU

logger = logging.getLogger(__name__)


class CPUGovernor:
    """
    Giám sát CPU usage và throttle khi vượt ngưỡng.

    Example::

        gov = CPUGovernor()
        gov.start()
        gov.throttle_if_needed(is_active=False)  # Background mode ≤40%
        gov.throttle_if_needed(is_active=True)    # Active mode ≤70%
    """

    def __init__(self) -> None:
        self._cpu_percent: float = 0.0
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        try:
            import psutil
            self._process = psutil.Process(os.getpid())
            self._has_psutil = True
        except ImportError:
            self._process = None
            self._has_psutil = False
            logger.warning("psutil not available. CPU monitoring disabled.")

    def start(self) -> None:
        """Bắt đầu monitoring thread."""
        if not self._has_psutil:
            return
        self._running.set()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="CPUGovernor", daemon=True,
        )
        self._monitor_thread.start()
        logger.info("CPU Governor started (interval=%.1fs).", CPU.monitor_interval)

    def stop(self) -> None:
        """Dừng monitoring."""
        self._running.clear()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    @property
    def cpu_percent(self) -> float:
        with self._lock:
            return self._cpu_percent

    def throttle_if_needed(self, is_active: bool = False) -> None:
        """
        Throttle (sleep) nếu CPU vượt ngưỡng.

        Args:
            is_active: True = Consumer đang inference (≤70%).
                       False = Background listening only (≤40%).
        """
        if not self._has_psutil:
            return
        cpu = self.cpu_percent
        if is_active:
            if cpu > CPU.active_max_percent:
                time.sleep(CPU.throttle_sleep_active)
        else:
            if cpu > CPU.background_max_percent:
                time.sleep(CPU.throttle_sleep_bg)

    def _monitor_loop(self) -> None:
        while self._running.is_set():
            try:
                cpu = self._process.cpu_percent(interval=CPU.monitor_interval)
                with self._lock:
                    self._cpu_percent = cpu
            except Exception:
                pass

    def get_system_info(self) -> dict:
        """Lấy thông tin hệ thống (debug/benchmark)."""
        info = {"cpu_percent": self.cpu_percent}
        if self._has_psutil:
            import psutil
            mem = self._process.memory_info()
            info.update({
                "rss_mb": mem.rss / (1024 * 1024),
                "vms_mb": mem.vms / (1024 * 1024),
                "threads": self._process.num_threads(),
                "system_cpu_count": psutil.cpu_count(),
            })
        return info
