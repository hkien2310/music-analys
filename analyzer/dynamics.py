"""
dynamics.py — Phân tích energy, loudness, dynamics theo thời gian.

Tính RMS stats, dynamic range (dB), LUFS loudness (nếu có pyloudnorm),
và energy timeline chart.
"""

import numpy as np
import librosa

from .config import DEPS
from .utils import progress, fmt_time

# Import pyloudnorm nếu có
try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

# ─── MODULE 1G: ENERGY & DYNAMICS ────────────────────────────────────────────

def analyze_dynamics(y, sr, duration):
    """Phân tích energy, loudness, dynamics theo thời gian"""
    progress("Phân tích dynamics và loudness...")
    result = {}

    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    result["rms_avg"]   = round(float(np.mean(rms)), 4)
    result["rms_max"]   = round(float(np.max(rms)), 4)
    result["rms_min"]   = round(float(np.min(rms)), 4)
    result["rms_std"]   = round(float(np.std(rms)), 4)
    # Clamp rms_min to avoid absurd dB values (silence → near-0 → 160+ dB)
    rms_min_safe = max(float(np.min(rms)), 0.001)
    dr = float(20 * np.log10(result["rms_max"] / rms_min_safe))
    result["dynamic_range_db"] = round(min(dr, 80.0), 1)  # Cap at 80 dB

    # Loudness chuẩn LUFS
    if DEPS["pyloudnorm"]:
        try:
            meter   = pyln.Meter(sr)
            loudness = meter.integrated_loudness(y.reshape(-1, 1) if y.ndim == 1 else y.T)
            result["loudness_lufs"] = round(float(loudness), 1)
            
            # So sánh với chuẩn streaming
            if loudness > -8:
                result["loudness_note"] = "RẤT TO — vượt chuẩn streaming, cần nén"
            elif loudness > -12:
                result["loudness_note"] = "TO — gần chuẩn, ổn"
            elif loudness > -16:
                result["loudness_note"] = "TRUNG BÌNH — đúng chuẩn Spotify (-14 LUFS)"
            elif loudness > -20:
                result["loudness_note"] = "NHẸ — hơi nhỏ so với chuẩn"
            else:
                result["loudness_note"] = "RẤT NHẸ — cần tăng loudness trước khi upload"
        except Exception:
            result["loudness_lufs"] = None
    else:
        result["loudness_lufs"] = None

    # Tạo energy timeline (ASCII chart theo 10 đoạn đều nhau)
    n_chunks = 20
    chunk_size = len(rms) // n_chunks
    timeline = []
    for i in range(n_chunks):
        chunk_rms = rms[i*chunk_size : (i+1)*chunk_size]
        avg = float(np.mean(chunk_rms)) if len(chunk_rms) > 0 else 0.0
        t   = duration * i / n_chunks
        timeline.append({"time": round(t, 1), "rms": round(avg, 4)})
    result["energy_timeline"] = timeline

    return result
