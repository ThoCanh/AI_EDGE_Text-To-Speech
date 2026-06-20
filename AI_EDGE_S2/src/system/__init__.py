"""
System module - Giám sát và kiểm soát tài nguyên hệ thống.

Exports:
    CPUGovernor: Giám sát CPU + adaptive throttle.
"""

from .cpu_governor import CPUGovernor

__all__ = ["CPUGovernor"]
