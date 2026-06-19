"""
memory_manager.py — Quản lý bộ nhớ và /dev/shm/ cho AI Edge Pipeline
=====================================================================
Khối 3 bài test: Memory Management

Nhiệm vụ:
- Đảm bảo model chỉ load 1 lần (Warm-up pattern)
- Giám sát RSS để phát hiện memory leak
- Quản lý file tạm trên /dev/shm/ (tmpfs)
- Dọn dẹp buffer trung gian sau mỗi inference cycle
"""

import os
import gc
import time
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Quản lý bộ nhớ trên thiết bị nhúng ARM.
    
    Chịu trách nhiệm:
    1. Giám sát RSS (Resident Set Size) phát hiện memory leak
    2. Quản lý /dev/shm/ cho file tạm TTS
    3. Force garbage collection sau mỗi inference cycle
    
    ═══════════════════════════════════════════════════════════
    DEEP DIVE: Tại sao /dev/shm/ thay vì /tmp/ ?
    ═══════════════════════════════════════════════════════════
    
    Trên Raspberry Pi 5 với MicroSD/SSD:
    
    ┌──────────────┬──────────────┬───────────────┬──────────────┐
    │ Thư mục      │ Backend      │ Tốc độ R/W    │ Hao mòn SD?  │
    ├──────────────┼──────────────┼───────────────┼──────────────┤
    │ /dev/shm/    │ tmpfs (RAM)  │ ~4-8 GB/s     │ KHÔNG        │
    │ /tmp/        │ ext4 hoặc    │ ~50 MB/s (SD) │ PHỤ THUỘC    │
    │              │ tmpfs (tùy)  │ ~4 GB/s (RAM) │              │
    │ /home/pi/    │ ext4 (SD)    │ ~50 MB/s      │ CÓ           │
    │ /var/tmp/    │ ext4 (SD)    │ ~50 MB/s      │ CÓ           │
    └──────────────┴──────────────┴───────────────┴──────────────┘
    
    /dev/shm/ là POSIX shared memory — luôn được mount là tmpfs 
    trên mọi distro Linux hiện đại. Điều này khác với /tmp/ vì:
    
    - Raspberry Pi OS: /tmp/ = tmpfs ✅ (mặc định)
    - Ubuntu Server:   /tmp/ = ext4 trên disk ❌
    - Debian minimal:  /tmp/ = ext4 trên disk ❌
    
    → /dev/shm/ là lựa chọn AN TOÀN NHẤT, không phụ thuộc distro.
    
    Với ứng dụng robot, MicroSD chịu hàng triệu write cycles/ngày 
    nếu ghi file âm thanh tạm. /dev/shm/ loại bỏ hoàn toàn vấn đề 
    wear leveling → kéo dài tuổi thọ phần cứng.
    
    Dung lượng mặc định: 50% RAM = 2GB (Pi5 4GB) hoặc 4GB (Pi5 8GB)
    → Dư sức cho file audio tạm (thường < 1MB/file)
    """

    def __init__(
        self,
        tmpfs_dir: str = "/dev/shm",
        check_interval: float = 30.0,
        leak_threshold_mb: float = 10.0,
    ):
        """
        Khởi tạo Memory Manager.
        
        Args:
            tmpfs_dir: Thư mục tmpfs để lưu file tạm (mặc định /dev/shm/)
            check_interval: Khoảng cách giữa các lần kiểm tra memory (giây)
            leak_threshold_mb: Ngưỡng cảnh báo memory leak (MB)
        """
        self._tmpfs_dir = tmpfs_dir
        self._check_interval = check_interval
        self._leak_threshold_mb = leak_threshold_mb

        # Baseline RSS khi khởi tạo (sau khi load model)
        self._baseline_rss_mb: Optional[float] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Danh sách file tạm đã tạo (để cleanup khi shutdown)
        self._temp_files: list[str] = []

        # Validate tmpfs directory
        self._validate_tmpfs()

    def _validate_tmpfs(self):
        """
        Kiểm tra /dev/shm/ tồn tại và writable.
        Fallback sang /tmp/ nếu không có (dev environment).
        """
        if os.path.isdir(self._tmpfs_dir) and os.access(self._tmpfs_dir, os.W_OK):
            logger.info(
                f"[MEM] ✅ tmpfs directory: {self._tmpfs_dir} "
                f"(size: {self._get_tmpfs_size_mb():.0f}MB available)"
            )
        else:
            import tempfile
            self._tmpfs_dir = tempfile.gettempdir()
            logger.warning(
                f"[MEM] ⚠️ /dev/shm/ không khả dụng, fallback: {self._tmpfs_dir}"
            )

    def _get_tmpfs_size_mb(self) -> float:
        """Lấy dung lượng khả dụng của tmpfs (MB)."""
        try:
            statvfs = os.statvfs(self._tmpfs_dir)
            return (statvfs.f_frsize * statvfs.f_bavail) / (1024 * 1024)
        except (OSError, AttributeError):
            return -1.0  # Windows không hỗ trợ statvfs

    def get_rss_mb(self) -> float:
        """
        Lấy RSS (Resident Set Size) hiện tại của process (MB).
        
        RSS = lượng RAM vật lý process đang chiếm.
        Nếu RSS tăng dần theo thời gian → memory leak.
        """
        try:
            # Linux: đọc /proc/self/status
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # VmRSS:    123456 kB
                        return int(line.split()[1]) / 1024.0
        except FileNotFoundError:
            pass

        # Fallback: dùng resource module (Unix) hoặc psutil
        try:
            import resource
            # ru_maxrss trả về KB trên Linux
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except ImportError:
            pass

        # Windows fallback
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return -1.0

    def set_baseline(self):
        """
        Đặt baseline RSS sau khi load xong tất cả model.
        Gọi hàm này SAU __init__() của VoicePipeline.
        
        Baseline = RSS sau warm-up = mức "bình thường".
        Mọi tăng trưởng RSS vượt baseline + threshold → leak.
        """
        self._baseline_rss_mb = self.get_rss_mb()
        logger.info(
            f"[MEM] 📊 Baseline RSS: {self._baseline_rss_mb:.1f}MB "
            f"(threshold: +{self._leak_threshold_mb}MB)"
        )

    def start_monitoring(self):
        """
        Bắt đầu background thread giám sát memory.
        Kiểm tra RSS mỗi {check_interval} giây.
        """
        if self._baseline_rss_mb is None:
            self.set_baseline()

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="MemoryMonitor",
            daemon=True,  # Tự tắt khi main thread kết thúc
        )
        self._monitor_thread.start()
        logger.info("[MEM] 🔍 Memory monitor đã khởi động")

    def _monitor_loop(self):
        """Background loop kiểm tra memory leak."""
        while not self._stop_event.is_set():
            current_rss = self.get_rss_mb()
            if current_rss > 0 and self._baseline_rss_mb is not None:
                delta = current_rss - self._baseline_rss_mb
                if delta > self._leak_threshold_mb:
                    logger.error(
                        f"[MEM] 🚨 MEMORY LEAK DETECTED! "
                        f"RSS: {current_rss:.1f}MB "
                        f"(+{delta:.1f}MB từ baseline)"
                    )
                else:
                    logger.debug(
                        f"[MEM] RSS: {current_rss:.1f}MB (Δ{delta:+.1f}MB)"
                    )
            self._stop_event.wait(self._check_interval)

    def stop_monitoring(self):
        """Dừng background monitor thread."""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

    # ─── /dev/shm/ File Operations ────────────────────────────

    def get_temp_path(self, filename: str) -> str:
        """
        Tạo đường dẫn file tạm trên /dev/shm/.
        
        Args:
            filename: Tên file (vd: "tts_output.raw")
            
        Returns:
            Full path trên tmpfs (vd: "/dev/shm/voice_pipeline_tts_output.raw")
        
        Tại sao prefix "voice_pipeline_":
        - /dev/shm/ là shared giữa tất cả processes
        - Prefix tránh xung đột tên file với process khác
        """
        prefixed = f"voice_pipeline_{filename}"
        path = os.path.join(self._tmpfs_dir, prefixed)
        self._temp_files.append(path)
        return path

    def write_temp(self, filename: str, data: bytes) -> str:
        """
        Ghi data vào file tạm trên /dev/shm/.
        
        Tốc độ: ~4-8 GB/s (RAM speed) vs ~50 MB/s (SD card)
        → Gần như tức thời cho file audio < 1MB
        """
        path = self.get_temp_path(filename)
        with open(path, "wb") as f:
            f.write(data)
        logger.debug(f"[MEM] 💾 Wrote {len(data)} bytes → {path}")
        return path

    def read_temp(self, filename: str) -> bytes:
        """Đọc file tạm từ /dev/shm/."""
        path = os.path.join(self._tmpfs_dir, f"voice_pipeline_{filename}")
        with open(path, "rb") as f:
            return f.read()

    def cleanup_temp(self, filename: Optional[str] = None):
        """
        Xóa file tạm trên /dev/shm/.
        Nếu filename=None, xóa tất cả file tạm đã tạo.
        """
        if filename:
            path = os.path.join(self._tmpfs_dir, f"voice_pipeline_{filename}")
            self._safe_remove(path)
        else:
            for path in self._temp_files:
                self._safe_remove(path)
            self._temp_files.clear()

    @staticmethod
    def _safe_remove(path: str):
        """Xóa file nếu tồn tại, không raise exception."""
        try:
            os.remove(path)
        except OSError:
            pass

    # ─── Garbage Collection ───────────────────────────────────

    @staticmethod
    def force_gc():
        """
        Force garbage collection sau mỗi inference cycle.
        
        Trên thiết bị nhúng với RAM giới hạn (4GB), Python's GC
        thresholds mặc định (700, 10, 10) có thể không đủ aggressive.
        
        Gọi gc.collect() tường minh đảm bảo:
        1. numpy arrays trung gian (PCM buffers) được giải phóng
        2. RSS không tăng dần (no memory leak)
        3. Headroom cho inference cycle tiếp theo
        """
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"[MEM] 🗑️ GC collected {collected} objects")

    # ─── Lifecycle ────────────────────────────────────────────

    def shutdown(self):
        """Cleanup tất cả resources khi tắt chương trình."""
        self.stop_monitoring()
        self.cleanup_temp()
        self.force_gc()
        logger.info("[MEM] 🔴 Memory manager shutdown complete")

    def get_status(self) -> dict:
        """Trả về trạng thái memory hiện tại."""
        rss = self.get_rss_mb()
        return {
            "rss_mb": rss,
            "baseline_mb": self._baseline_rss_mb,
            "delta_mb": (rss - self._baseline_rss_mb) if self._baseline_rss_mb else 0,
            "tmpfs_dir": self._tmpfs_dir,
            "tmpfs_available_mb": self._get_tmpfs_size_mb(),
            "temp_files_count": len(self._temp_files),
        }
