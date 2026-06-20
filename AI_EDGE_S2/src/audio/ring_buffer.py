"""
Ring Buffer - Bộ đệm vòng kích thước cố định cho audio stream.

Sử dụng collections.deque(maxlen=N) để:
- Tự động discard frame cũ nhất khi đầy (O(1) amortized)
- TUYỆT ĐỐI không gây memory leak sau vài giờ hoạt động
- Thread-safe qua threading.Lock()

Đây là yêu cầu bắt buộc của Khối 1: Không dùng list/array vô hạn.
"""

from __future__ import annotations

import collections
import threading
from typing import Optional

import numpy as np

from ..config import AUDIO


class RingBuffer:
    """
    Bộ đệm vòng (circular buffer) cho dữ liệu âm thanh PCM.

    Đặc điểm:
        - Kích thước cố định: tối đa ``AUDIO.buffer_seconds`` giây.
        - Thread-safe: an toàn khi Thread 1 ghi, Thread 2 đọc.
        - Zero memory leak: deque(maxlen) tự động loại bỏ phần tử cũ.

    Example::

        buf = RingBuffer()
        buf.write(np.zeros(512, dtype=np.float32))
        audio = buf.read_all()
        pre_roll = buf.read_last_n_ms(300)  # 300ms gần nhất
    """

    __slots__ = ("_buffer", "_lock", "_chunk_ms")

    def __init__(
        self,
        max_chunks: Optional[int] = None,
        chunk_ms: int = AUDIO.chunk_ms,
    ) -> None:
        self._chunk_ms = chunk_ms
        capacity = max_chunks if max_chunks else AUDIO.max_buffer_chunks
        self._buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=capacity,
        )
        self._lock = threading.Lock()

    # ─────────────────────────────────────────
    # Ghi dữ liệu (Producer thread gọi)
    # ─────────────────────────────────────────

    def write(self, chunk: np.ndarray) -> None:
        """
        Ghi 1 chunk PCM vào buffer.

        Khi buffer đầy, chunk cũ nhất bị loại tự động.
        Copy dữ liệu để tránh reference tới numpy buffer ngoài.

        Args:
            chunk: Mảng numpy float32, kích thước = AUDIO.chunk_size.
        """
        with self._lock:
            self._buffer.append(chunk.copy())

    # ─────────────────────────────────────────
    # Đọc dữ liệu (Consumer thread gọi)
    # ─────────────────────────────────────────

    def read_all(self) -> np.ndarray:
        """
        Đọc toàn bộ nội dung buffer thành 1 mảng liên tục.

        Returns:
            Mảng numpy float32. Rỗng nếu buffer chưa có data.
        """
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buffer))

    def read_last_n_ms(self, ms: int) -> np.ndarray:
        """
        Đọc N mili-giây gần nhất từ buffer (pre-roll).

        Dùng để prepend audio trước thời điểm VAD detect speech,
        tránh cắt mất phần đầu của câu nói.

        Args:
            ms: Số mili-giây cần đọc (ví dụ: 300ms).

        Returns:
            Mảng numpy float32 chứa audio pre-roll.
        """
        n_chunks = max(1, int(ms / self._chunk_ms))
        with self._lock:
            chunks = list(self._buffer)[-n_chunks:]
            if not chunks:
                return np.array([], dtype=np.float32)
            return np.concatenate(chunks)

    # ─────────────────────────────────────────
    # Tiện ích
    # ─────────────────────────────────────────

    def clear(self) -> None:
        """Xóa toàn bộ buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def duration_ms(self) -> float:
        """Thời lượng audio hiện tại trong buffer (ms)."""
        return len(self._buffer) * self._chunk_ms

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self._buffer.maxlen

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"RingBuffer(chunks={len(self._buffer)}/{self._buffer.maxlen}, "
            f"duration={self.duration_ms:.0f}ms)"
        )
