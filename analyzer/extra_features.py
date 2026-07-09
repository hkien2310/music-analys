"""
extra_features.py — Các đặc trưng bổ sung: danceability estimate, valence estimate.

Danceability dựa trên beat regularity + energy.
Valence estimate từ spectral brightness + tempo (heuristic).
"""

import numpy as np
import librosa

from .utils import progress

# ─── MODULE 1I: EXTRA FEATURES ────────────────────────────────────────────────

def analyze_extra(y, sr, duration):
    """Các đặc trưng bổ sung: danceability estimate, valence estimate"""
    progress("Phân tích đặc trưng bổ sung...")
    result = {}

    hop_length = 512

    # Estimate danceability từ regularity of beats + energy
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    
    # Beat regularity (1 = perfect metronome, 0 = hoàn toàn không đều)
    try:
        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        if len(beats) > 4:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            ibis = np.diff(beat_times)
            regularity = 1.0 - float(np.std(ibis) / (np.mean(ibis) + 1e-9))
            regularity = max(0.0, min(1.0, regularity))
        else:
            regularity = 0.5
    except Exception:
        regularity = 0.5

    # Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    energy_norm = float(np.mean(rms)) / 0.15  # normalize thực nghiệm
    energy_norm = max(0.0, min(1.0, energy_norm))

    # Danceability estimate
    danceability = (regularity * 0.6 + energy_norm * 0.4)
    danceability = round(max(0.0, min(1.0, danceability)), 2)
    result["danceability_score"] = danceability
    if danceability >= 0.75:
        result["danceability_label"] = "Rất cao — rất phù hợp để nhảy/dance"
    elif danceability >= 0.55:
        result["danceability_label"] = "Cao — phù hợp dance"
    elif danceability >= 0.35:
        result["danceability_label"] = "Trung bình"
    else:
        result["danceability_label"] = "Thấp — không phù hợp để dance"

    # Valence estimate từ spectral brightness + scale
    # (heuristic: bài sáng & nhanh thường "vui hơn")
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    brightness_norm = min(1.0, spec_cent / 4000.0)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    tempo_norm = min(1.0, max(0.0, (tempo_val - 60) / 120.0))

    valence = round(brightness_norm * 0.5 + tempo_norm * 0.5, 2)
    result["valence_score"] = valence
    if valence >= 0.7:
        result["valence_label"] = "Tươi sáng / Vui tươi"
    elif valence >= 0.5:
        result["valence_label"] = "Trung tính / Nhẹ nhàng"
    elif valence >= 0.35:
        result["valence_label"] = "Sâu lắng / Trầm tư"
    else:
        result["valence_label"] = "Tối / Buồn / Melancholic"

    return result
