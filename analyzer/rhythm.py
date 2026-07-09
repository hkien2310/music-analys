"""
rhythm.py — Phân tích nhịp điệu (BPM, beats, time signature).

Sử dụng librosa để phát hiện tempo, beat positions, time signature,
tempo stability, và onset density.
"""

import numpy as np
import librosa

from .utils import progress

# ─── MODULE 1C: RHYTHM ────────────────────────────────────────────────────────

def analyze_rhythm(y, sr):
    """Phân tích nhịp: BPM, beats, time signature"""
    progress("Phân tích nhịp điệu (BPM, beats)...")
    result = {}

    # Tempo & beat positions
    onset_env    = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Xử lý tempo có thể là array
    if hasattr(tempo, '__len__'):
        tempo_val = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo_val = float(tempo)

    result["tempo_bpm"]        = round(tempo_val, 1)
    result["beats_total"]      = int(len(beats))
    result["beat_times"]       = librosa.frames_to_time(beats, sr=sr).tolist()

    # Ước tính time signature từ tempogram
    try:
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        # Thử detect downbeats bằng cách tìm mẫu năng lượng
        # Nếu energy mỗi nhịp 1,3,5... < nhịp 2,4,6... → likely 3/4
        # Đây là heuristic đơn giản
        if len(beats) > 8:
            beat_energies = []
            for b in beats[:min(32, len(beats))]:
                start = max(0, b - 2)
                end   = min(len(onset_env) - 1, b + 2)
                beat_energies.append(float(np.mean(onset_env[start:end+1])))
            
            # Tìm độ lệch năng lượng giữa các nhịp
            # Pattern 4/4: strong-weak-medium-weak
            # Pattern 3/4: strong-weak-weak
            # Dùng autocorrelation để phát hiện
            if len(beat_energies) >= 8:
                corr3 = sum(beat_energies[i] * beat_energies[i+3]
                           for i in range(len(beat_energies)-3)) / (len(beat_energies)-3)
                corr4 = sum(beat_energies[i] * beat_energies[i+4]
                           for i in range(len(beat_energies)-4)) / (len(beat_energies)-4)
                if corr3 > corr4 * 1.1:
                    result["time_signature"] = "3/4"
                    result["time_sig_confidence"] = "Trung bình"
                else:
                    result["time_signature"] = "4/4"
                    result["time_sig_confidence"] = "Cao"
            else:
                result["time_signature"] = "4/4"
                result["time_sig_confidence"] = "Thấp (ít beats)"
        else:
            result["time_signature"] = "4/4"
            result["time_sig_confidence"] = "Thấp (ít beats)"
    except Exception:
        result["time_signature"] = "4/4"
        result["time_sig_confidence"] = "Không xác định"

    # Tempo stability
    if len(result["beat_times"]) > 4:
        ibis = np.diff(result["beat_times"])
        result["tempo_stability"] = round(float(1 - np.std(ibis) / np.mean(ibis)), 3)
    else:
        result["tempo_stability"] = 0.0

    # Onset density (events per second)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr)
    duration     = librosa.get_duration(y=y, sr=sr)
    result["onset_density_per_sec"] = round(float(len(onset_times)) / duration, 2) if duration > 0 else 0.0

    return result
