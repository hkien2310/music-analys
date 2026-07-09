"""
utils.py — Các hàm tiện ích dùng chung trong analyzer package.

Bao gồm:
- fmt_time: Định dạng giây sang M:SS
- progress: In trạng thái tiến trình
- ascii_bar: Tạo thanh ASCII bar
- normalize_chord_name: Tạo tên hợp âm đẹp
- cosine_sim: Tính cosine similarity giữa 2 vector
"""

import numpy as np

from .config import PITCH_CLASSES

# ─── UTILITY ──────────────────────────────────────────────────────────────────

def fmt_time(seconds):
    """Định dạng giây sang M:SS"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"

def progress(msg):
    print(f"  → {msg}", flush=True)

def ascii_bar(value, max_val, width=20, char="█"):
    """Tạo thanh ASCII bar"""
    filled = int(round(value / max_val * width)) if max_val > 0 else 0
    filled = max(0, min(filled, width))
    return char * filled + "░" * (width - filled)

def normalize_chord_name(root_idx, quality):
    """Tạo tên hợp âm đẹp"""
    root = PITCH_CLASSES[root_idx % 12]
    return f"{root}{quality}"

def cosine_sim(a, b):
    """Tính cosine similarity giữa 2 vector"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
