"""
CPU Governor - Giám sát và kiểm soát tài nguyên CPU.

Đảm bảo tuân thủ ràng buộc phần cứng (đề bài):
    - Background (Audio + VAD): ≤ 40% CPU
    - Active (ASR + TTS):       ≤ 70% CPU

Chiến lược 3 lớp:
    Lớp 1: ONNX thread pinning (VAD=1 thread, ASR=2 threads)
    Lớp 2: Adaptive throttle với exponential backoff
    Lớp 3: Stats tracking + warning log khi vi phạm liên tục

Tại sao cần CPU Governor?
    Trên Pi5, pipeline voice assistant chia sẻ CPU với motor control
    và dashboard display. Nếu ASR/TTS chiếm hết CPU → motor mất
    kiểm soát → nguy hiểm. CPU Governor đảm bảo voice pipeline
    luôn "nhường" CPU cho các tiến trình quan trọng hơn.
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

    Chiến lược Adaptive Throttle:
        1. Đo CPU usage liên tục qua monitor thread (mỗi 1s)
        2. Producer gọi throttle_if_needed() sau mỗi chunk 32ms
        3. Nếu CPU vượt ngưỡng → sleep adaptive (10ms → 20ms → 40ms)
        4. Nếu CPU về dưới ngưỡng → reset sleep về 10ms

    Example::

        gov = CPUGovernor()
        gov.start()
        gov.throttle_if_needed(is_active=False)  # Background mode ≤40%
        gov.throttle_if_needed(is_active=True)    # Active mode ≤70%
    """

    # Adaptive backoff: sleep tăng dần khi vi phạm liên tục
    _MIN_SLEEP_BG: float = 0.005         # 5ms (bắt đầu nhẹ)
    _MAX_SLEEP_BG: float = 0.05          # 50ms (tối đa)
    _MIN_SLEEP_ACTIVE: float = 0.002     # 2ms
    _MAX_SLEEP_ACTIVE: float = 0.02      # 20ms
    _BACKOFF_FACTOR: float = 1.5         # Tăng 1.5x mỗi lần vi phạm

    def __init__(self) -> None:
        self._cpu_percent: float = 0.0
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        # Adaptive sleep state
        self._current_sleep_bg: float = self._MIN_SLEEP_BG
        self._current_sleep_active: float = self._MIN_SLEEP_ACTIVE

        # Stats tracking
        self._stats = {
            "bg_throttle_count": 0,       # Số lần throttle ở background
            "active_throttle_count": 0,    # Số lần throttle ở active
            "bg_violations": 0,            # Số lần CPU vượt 40% (background)
            "active_violations": 0,        # Số lần CPU vượt 70% (active)
            "peak_cpu_percent": 0.0,       # CPU cao nhất ghi nhận
        }

        try:
            import psutil
            self._process = psutil.Process(os.getpid())
            self._has_psutil = True
        except ImportError:
            self._process = None
            self._has_psutil = False
            logger.warning("psutil not available. CPU monitoring disabled.")

    # ═══════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════

    def start(self) -> None:
        """Bắt đầu monitoring thread."""
        if not self._has_psutil:
            return
        self._running.set()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="CPUGovernor", daemon=True,
        )
        self._monitor_thread.start()
        logger.info(
            "CPU Governor started. Budget: BG≤%.0f%%, Active≤%.0f%%, interval=%.1fs",
            CPU.background_max_percent, CPU.active_max_percent,
            CPU.monitor_interval,
        )

    def stop(self) -> None:
        """Dừng monitoring. Log stats cuối cùng."""
        self._running.clear()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("CPU Governor stopped. Stats: %s", self._stats)

    # ═══════════════════════════════════════════════════════
    # CPU Throttle (gọi từ Producer và Consumer threads)
    # ═══════════════════════════════════════════════════════

    @property
    def cpu_percent(self) -> float:
        with self._lock:
            return self._cpu_percent

    def throttle_if_needed(self, is_active: bool = False) -> None:
        """
        Throttle (sleep) nếu CPU vượt ngưỡng.

        Chiến lược Adaptive Backoff:
            - Lần vi phạm đầu: sleep MIN_SLEEP (5ms bg / 2ms active)
            - Vi phạm tiếp: sleep × 1.5 (exponential backoff)
            - CPU về bình thường: reset sleep về MIN
            - Cap tại MAX_SLEEP để không ảnh hưởng real-time

        Args:
            is_active: True = Consumer đang inference (ASR+TTS ≤70%).
                       False = Background listening only (Audio+VAD ≤40%).
        """
        if not self._has_psutil:
            return

        cpu = self.cpu_percent

        if is_active:
            if cpu > CPU.active_max_percent:
                # ── VƯỢT 70% → throttle ──────────────────
                time.sleep(self._current_sleep_active)
                self._stats["active_throttle_count"] += 1
                self._stats["active_violations"] += 1
                # Escalate: sleep dài hơn nếu vẫn vi phạm
                self._current_sleep_active = min(
                    self._current_sleep_active * self._BACKOFF_FACTOR,
                    self._MAX_SLEEP_ACTIVE,
                )
            else:
                # CPU về bình thường → reset adaptive sleep
                self._current_sleep_active = self._MIN_SLEEP_ACTIVE
        else:
            if cpu > CPU.background_max_percent:
                # ── VƯỢT 40% → throttle ──────────────────
                time.sleep(self._current_sleep_bg)
                self._stats["bg_throttle_count"] += 1
                self._stats["bg_violations"] += 1
                # Escalate: sleep dài hơn nếu vẫn vi phạm
                self._current_sleep_bg = min(
                    self._current_sleep_bg * self._BACKOFF_FACTOR,
                    self._MAX_SLEEP_BG,
                )
            else:
                # CPU về bình thường → reset adaptive sleep
                self._current_sleep_bg = self._MIN_SLEEP_BG

        # Track peak
        if cpu > self._stats["peak_cpu_percent"]:
            self._stats["peak_cpu_percent"] = cpu

    def throttle_consumer(self) -> None:
        """
        Throttle dành riêng cho Consumer thread (ASR+TTS).

        Gọi GIỮA ASR và TTS inference để kiểm tra CPU
        trước khi bắt đầu task nặng tiếp theo.
        """
        self.throttle_if_needed(is_active=True)

    # ═══════════════════════════════════════════════════════
    # Monitoring
    # ═══════════════════════════════════════════════════════

    def _monitor_loop(self) -> None:
        """
        Thread giám sát CPU liên tục.

        Đo CPU usage mỗi monitor_interval giây.
        Log warning nếu vi phạm liên tục > 10 lần.
        """
        consecutive_violations = 0

        while self._running.is_set():
            try:
                cpu = self._process.cpu_percent(interval=CPU.monitor_interval)
                with self._lock:
                    self._cpu_percent = cpu

                # Warning nếu vi phạm liên tục
                if cpu > CPU.active_max_percent:
                    consecutive_violations += 1
                    if consecutive_violations >= 10:
                        logger.warning(
                            "[CPUGovernor] ⚠️ CPU=%.1f%% > %.0f%% "
                            "for %d consecutive checks!",
                            cpu, CPU.active_max_percent,
                            consecutive_violations,
                        )
                        consecutive_violations = 0
                else:
                    consecutive_violations = 0

            except Exception:
                pass

    # ═══════════════════════════════════════════════════════
    # Debug / Benchmark
    # ═══════════════════════════════════════════════════════

    @property
    def stats(self) -> dict:
        """Trả về CPU throttle stats."""
        return {**self._stats, "current_cpu_percent": self.cpu_percent}

    def get_system_info(self) -> dict:
        """Lấy thông tin hệ thống (debug/benchmark)."""
        info = {
            "cpu_percent": self.cpu_percent,
            "cpu_budget": {
                "background_limit": CPU.background_max_percent,
                "active_limit": CPU.active_max_percent,
            },
            "throttle_stats": self._stats.copy(),
        }
        if self._has_psutil:
            import psutil
            mem = self._process.memory_info()
            info.update({
                "rss_mb": round(mem.rss / (1024 * 1024), 2),
                "vms_mb": round(mem.vms / (1024 * 1024), 2),
                "threads": self._process.num_threads(),
                "system_cpu_count": psutil.cpu_count(),
                "system_cpu_freq_mhz": (
                    psutil.cpu_freq().current if psutil.cpu_freq() else None
                ),
            })
        return info
