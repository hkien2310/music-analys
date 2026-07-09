"""
key_detection.py — Phát hiện tông nhạc (Key/Scale) với confidence score.

Sử dụng CQT Chroma và profile Krumhansl-Schmuckler để detect key,
xếp hạng ứng viên, và phát hiện relative key.
"""

import numpy as np
import librosa

from .config import PITCH_CLASSES, KS_MAJOR, KS_MINOR
from .utils import progress

# ─── MODULE 1D: KEY & SCALE ───────────────────────────────────────────────────

def analyze_key(y, sr):
    """Phát hiện tông nhạc với confidence score"""
    progress("Phân tích tông nhạc (Key/Scale)...")

    # CQT Chroma - chính xác hơn STFT chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    mean_chroma = np.mean(chroma, axis=1)
    mean_chroma = mean_chroma / (np.sum(mean_chroma) + 1e-9)  # normalize

    best_key   = 0
    best_scale = "Major"
    best_corr  = -2.0
    all_scores = []

    for i in range(12):
        rot_major = np.roll(KS_MAJOR, i)
        rot_minor = np.roll(KS_MINOR, i)

        corr_maj = np.corrcoef(mean_chroma, rot_major / rot_major.sum())[0, 1]
        corr_min = np.corrcoef(mean_chroma, rot_minor / rot_minor.sum())[0, 1]

        all_scores.append((PITCH_CLASSES[i], "Major", float(corr_maj)))
        all_scores.append((PITCH_CLASSES[i], "Minor", float(corr_min)))

        if corr_maj > best_corr:
            best_corr = corr_maj; best_key = i; best_scale = "Major"
        if corr_min > best_corr:
            best_corr = corr_min; best_key = i; best_scale = "Minor"

    # Sort tất cả ứng viên
    all_scores.sort(key=lambda x: x[2], reverse=True)
    top5 = all_scores[:5]

    # Confidence: khoảng cách giữa #1 và #2
    if len(top5) >= 2:
        gap = top5[0][2] - top5[1][2]
        if gap > 0.15:   confidence_pct = 90
        elif gap > 0.08: confidence_pct = 75
        elif gap > 0.04: confidence_pct = 60
        else:            confidence_pct = 45
    else:
        confidence_pct = 50

    # Phát hiện relative key
    if best_scale == "Major":
        rel_root = (best_key + 9) % 12
        relative_key = f"{PITCH_CLASSES[rel_root]} Minor (Thứ song song)"
    else:
        rel_root = (best_key + 3) % 12
        relative_key = f"{PITCH_CLASSES[rel_root]} Major (Trưởng song song)"

    return {
        "root"          : PITCH_CLASSES[best_key],
        "scale"         : best_scale,
        "full"          : f"{PITCH_CLASSES[best_key]} {best_scale}",
        "confidence_pct": confidence_pct,
        "relative_key"  : relative_key,
        "top_candidates": [(n, s, round(c, 3)) for n, s, c in top5],
        "mean_chroma"   : mean_chroma.tolist(),
    }
