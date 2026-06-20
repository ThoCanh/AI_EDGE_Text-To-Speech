"""
Từ điển domain-specific cho xe điện / automotive.

Tách riêng từ điển ra khỏi logic normalizer để:
- Dễ mở rộng: thêm từ mới = thêm 1 dòng
- Dễ maintain: QA/domain expert có thể sửa trực tiếp
- Có thể load từ file JSON/YAML trong production
"""

from typing import Dict

# ═══════════════════════════════════════════════════════════
# Viết tắt kỹ thuật → phiên âm tiếng Việt
# ═══════════════════════════════════════════════════════════

ACRONYM_MAP: Dict[str, str] = {
    # Automotive / EV
    "BMS": "bi em ét",
    "CAN": "can",
    "ECU": "i xi du",
    "OBD": "ô bi đi",
    "ABS": "ây bi ét",
    "EPS": "i pi ét",
    "VCU": "vi xi du",
    "MCU": "em xi du",
    "HV": "ết vi",
    "LV": "eo vi",
    "SOC": "ét ô xi",
    "SOH": "ét ô ết",
    "OTA": "ô ti ây",
    # Electronics
    "GPS": "gi pi ét",
    "LED": "lét",
    "USB": "du ét bi",
    "CPU": "xi pi du",
    "RAM": "ram",
    "SoC": "ét ô xi",
    "PCB": "pi xi bi",
    "API": "ây pi ai",
    "IoT": "ai ô ti",
    "PWM": "pi đáp-liu em",
    "ADC": "ây đi xi",
    "DAC": "đi ây xi",
    "I2C": "ai tu xi",
    "SPI": "ét pi ai",
    "UART": "du a ti",
    "GPIO": "gi pi ai ô",
}

# ═══════════════════════════════════════════════════════════
# Thuật ngữ kỹ thuật tiếng Anh → phiên âm Việt hóa
# ═══════════════════════════════════════════════════════════

TECH_TERMS: Dict[str, str] = {
    # Lỗi / cảnh báo
    "overcurrent": "ô-vơ-ca-rần",
    "overvoltage": "ô-vơ-vôn-tít",
    "undervoltage": "ần-đơ-vôn-tít",
    "overheat": "ô-vơ-hít",
    "overload": "ô-vơ-lốt",
    "timeout": "thai-ao",
    "error": "e-rơ",
    "warning": "wo-ninh",
    "critical": "crít-ti-cồ",
    "fault": "phôn",
    "failure": "phây-liơ",
    "shutdown": "sát-đao",
    "emergency": "i-mơ-giần-xi",
    # Kỹ thuật
    "communication": "cơm-miu-ni-kây-sần",
    "controller": "cơn-trô-lơ",
    "voltage": "vôn-tít",
    "current": "ca-rần",
    "battery": "bát-tơ-ri",
    "firmware": "phơm-we",
    "software": "xóp-we",
    "hardware": "hát-we",
    "driver": "đrai-vơ",
    "sensor": "xen-xơ",
    "module": "mô-đun",
    "inverter": "in-vơ-tơ",
    "charger": "chác-giơ",
    "connector": "cơ-nếch-tơ",
    "dashboard": "đát-bót",
    "display": "đít-plây",
    "update": "áp-đết",
    "reset": "ri-xét",
    "reboot": "ri-bút",
    "bus": "bát",
    "protocol": "prô-tô-con",
}

# ═══════════════════════════════════════════════════════════
# Đơn vị đo (sắp xếp dài trước ngắn để tránh match sai)
# ═══════════════════════════════════════════════════════════

UNITS: Dict[str, str] = {
    "kWh": "ki-lô oát giờ",
    "kHz": "ki-lô héc",
    "MHz": "mê-ga héc",
    "mAh": "mi-li am-pe giờ",
    "kW": "ki-lô oát",
    "mA": "mi-li am-pe",
    "mV": "mi-li vôn",
    "°C": "độ xê",
    "rpm": "vòng trên phút",
    "km/h": "ki-lô-mét trên giờ",
    "km": "ki-lô-mét",
    "Ah": "am-pe giờ",
    "Hz": "héc",
    "V": "vôn",
    "A": "am-pe",
    "W": "oát",
}

# ═══════════════════════════════════════════════════════════
# Chữ cái → phiên âm tiếng Việt (cho fallback spell-out)
# ═══════════════════════════════════════════════════════════

LETTER_MAP: Dict[str, str] = {
    "a": "ây", "b": "bi", "c": "xi", "d": "đi", "e": "i",
    "f": "ép", "g": "gi", "h": "ết", "i": "ai", "j": "giây",
    "k": "kây", "l": "eo", "m": "em", "n": "en", "o": "ô",
    "p": "pi", "q": "kiu", "r": "a", "s": "ét", "t": "ti",
    "u": "du", "v": "vi", "w": "đáp-liu", "x": "ích",
    "y": "oai", "z": "dét",
}

# ═══════════════════════════════════════════════════════════
# Từ tiếng Việt KHÔNG DẤU phổ biến (passthrough, không spell-out)
# ═══════════════════════════════════════════════════════════

VIETNAMESE_PASSTHROUGH: frozenset = frozenset({
    # Từ phổ biến trong ngữ cảnh kỹ thuật
    "loi", "ma", "he", "thong", "dang", "kiem", "tra", "phat", "hien",
    "tren", "duong", "nguon", "do", "bi", "khong", "co", "the", "da",
    "va", "hoac", "khi", "trong", "ngoai", "trang", "thai", "hoat",
    "dong", "binh", "thuong", "nhiet", "pin", "xe", "may", "dien",
    "tu", "dong", "hoa", "canh", "bao", "nguy", "hiem", "mat",
    "ket", "noi", "phan", "hoi", "cap", "nhat", "cua", "cho",
    "voi", "den", "tu", "bang", "so", "ten", "ghi", "doc", "gui",
    "nhan", "bat", "tat", "mo", "dong", "len", "xuong",
})

