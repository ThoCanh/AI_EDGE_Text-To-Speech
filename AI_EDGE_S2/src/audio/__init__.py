"""
Audio I/O module - Thu âm và quản lý bộ đệm.

Exports:
    RingBuffer: Bộ đệm vòng kích thước cố định (Khối 1).
"""

from .ring_buffer import RingBuffer

__all__ = ["RingBuffer"]
