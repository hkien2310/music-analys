"""
timbre.py — Phân tích âm sắc và đặc trưng phổ.

Bao gồm: MFCCs, spectral centroid/bandwidth/rolloff/flatness,
zero crossing rate, HPSS (harmonic/percussive separation),
brightness/tonal quality assessment, và dominant notes.
"""

import numpy as np
import librosa

from .config import PITCH_CLASSES
from .utils import progress

# ─── MODULE 1H: TIMBRE & SPECTRAL ─────────────────────────────────────────────

def analyze_timbre(y, sr):
    """Phân tích âm sắc và đặc trưng phổ"""
    progress("Phân tích âm sắc và đặc trưng phổ...")
    result = {}

    hop_length = 512

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    result["mfcc_means"] = [round(float(m), 3) for m in np.mean(mfcc, axis=1)]

    # Spectral features
    spec_cent    = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spec_bw      = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    spec_flat    = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)
    zcr          = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)

    result["spectral_centroid_hz"]  = round(float(np.mean(spec_cent)), 1)
    result["spectral_bandwidth_hz"] = round(float(np.mean(spec_bw)), 1)
    result["spectral_rolloff_hz"]   = round(float(np.mean(spec_rolloff)), 1)
    result["spectral_flatness"]     = round(float(np.mean(spec_flat)), 4)
    result["zero_crossing_rate"]    = round(float(np.mean(zcr)), 4)

    # Diễn giải spectral centroid
    sc_hz = result["spectral_centroid_hz"]
    if sc_hz > 4000:
        result["brightness"] = "Rất sáng (Bright) — nhiều treble, hi-hat, cymbal"
    elif sc_hz > 2500:
        result["brightness"] = "Sáng (Bright) — cân bằng, vocal rõ"
    elif sc_hz > 1500:
        result["brightness"] = "Trung (Neutral) — cân bằng mid-range"
    elif sc_hz > 800:
        result["brightness"] = "Ấm (Warm) — nhiều mid-bass"
    else:
        result["brightness"] = "Tối (Dark) — bass nặng, lo-fi hoặc sub bass"

    # Diễn giải spectral flatness (noise vs tone)
    sf = result["spectral_flatness"]
    if sf > 0.3:
        result["tonal_quality"] = "Nhiều noise — percussive, distortion, white noise"
    elif sf > 0.1:
        result["tonal_quality"] = "Hỗn hợp — vừa tonal vừa percussive"
    else:
        result["tonal_quality"] = "Rất tonal — âm nhạc rõ ràng, ít noise"

    # HPSS - Harmonic/Percussive separation
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = float(np.sum(y_harm ** 2))
    perc_energy = float(np.sum(y_perc ** 2))
    total_energy = harm_energy + perc_energy + 1e-9

    result["harmonic_ratio"]   = round(harm_energy / total_energy, 3)
    result["percussive_ratio"] = round(perc_energy / total_energy, 3)

    if result["harmonic_ratio"] > 0.75:
        result["hp_description"] = "Chủ yếu giai điệu (>75% Harmonic) — nhạc melody chủ đạo"
    elif result["harmonic_ratio"] > 0.55:
        result["hp_description"] = "Cân bằng giai điệu-nhịp điệu"
    elif result["harmonic_ratio"] > 0.40:
        result["hp_description"] = "Thiên về nhịp điệu"
    else:
        result["hp_description"] = "Chủ yếu bộ gõ (>60% Percussive) — nhạc dance/electronic"

    # Chromagram summary
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    dominant_notes_idx = np.argsort(chroma_mean)[::-1][:4]
    result["dominant_notes"] = [PITCH_CLASSES[i] for i in dominant_notes_idx]

    return result
