"""
structure.py — Phát hiện cấu trúc bài nhạc (Intro/Verse/Chorus/Bridge/Outro).

Sử dụng RMS energy, MFCC, chroma, self-similarity matrix, và agglomerative
clustering để phân đoạn. Gán label dựa trên position + energy heuristics.
"""

import numpy as np
import librosa

from .utils import fmt_time, progress

# ─── MODULE 1F: SONG STRUCTURE ────────────────────────────────────────────────

def analyze_structure(y, sr, duration, beat_times):
    """Phát hiện cấu trúc bài nhạc (Intro/Verse/Chorus/Bridge/Outro)"""
    progress("Phân tích cấu trúc bài nhạc (Intro/Verse/Chorus/...)...")

    hop_length = 512
    
    # Tính RMS energy theo time frames
    rms       = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Tính MFCCs để làm feature cho segmentation
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # Concatenate features
    features = np.vstack([mfcc, chroma])

    # Self-similarity matrix
    try:
        R = librosa.segment.recurrence_matrix(
            features, width=3, mode='affinity', sym=True
        )
        
        # Tìm structural boundaries
        bounds = librosa.segment.agglomerative(features, k=min(8, max(3, int(duration // 30))))
        bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length).tolist()
    except Exception:
        # Fallback: chia đều theo energy
        n_segs = max(3, min(8, int(duration // 30)))
        bound_times = [duration * i / n_segs for i in range(n_segs + 1)]

    # Thêm 0 và duration nếu chưa có
    if not bound_times or bound_times[0] > 1.0:
        bound_times.insert(0, 0.0)
    if bound_times[-1] < duration - 1.0:
        bound_times.append(duration)

    # Loại bỏ boundary quá gần nhau (< 10 giây)
    filtered = [bound_times[0]]
    for t in bound_times[1:]:
        if t - filtered[-1] >= 10.0:
            filtered.append(t)
    if filtered[-1] < duration - 5:
        filtered.append(duration)
    bound_times = filtered

    # Tính đặc trưng cho từng segment
    segments = []
    global_rms_max = float(np.max(rms)) if len(rms) > 0 else 1.0

    for i in range(len(bound_times) - 1):
        t_start = bound_times[i]
        t_end   = bound_times[i + 1]
        
        # RMS trong segment này
        mask = (rms_times >= t_start) & (rms_times < t_end)
        seg_rms = rms[mask]
        avg_rms  = float(np.mean(seg_rms)) if len(seg_rms) > 0 else 0.0
        max_rms  = float(np.max(seg_rms))  if len(seg_rms) > 0 else 0.0
        rms_ratio = avg_rms / (global_rms_max + 1e-9)
        
        # Spectral centroid trong segment
        y_seg = y[int(t_start * sr):int(t_end * sr)]
        if len(y_seg) > 0:
            sc    = librosa.feature.spectral_centroid(y=y_seg, sr=sr)
            avg_sc = float(np.mean(sc))
        else:
            avg_sc = 0.0

        segments.append({
            "index"    : i + 1,
            "start"    : round(t_start, 2),
            "end"      : round(t_end, 2),
            "duration" : round(t_end - t_start, 2),
            "start_str": fmt_time(t_start),
            "end_str"  : fmt_time(t_end),
            "avg_rms"  : round(avg_rms, 4),
            "rms_ratio": round(rms_ratio, 3),
            "spec_cent": round(avg_sc, 1),
        })

    # Label thông minh dựa trên position + energy
    # Heuristic: Intro=đầu&nhẹ, Outro=cuối&nhẹ, Chorus=năng lượng cao, Verse=trung bình
    n = len(segments)
    rms_values = [s["rms_ratio"] for s in segments]
    rms_arr = np.array(rms_values)
    high_thresh = np.percentile(rms_arr, 70)
    low_thresh  = np.percentile(rms_arr, 35)
    
    labeled = []
    verse_count   = 0
    chorus_count  = 0
    bridge_count  = 0
    
    for i, seg in enumerate(segments):
        pos_ratio  = i / max(n - 1, 1)
        rms_ratio  = seg["rms_ratio"]

        # Determine label
        if i == 0 and rms_ratio < high_thresh:
            label = "Intro"
        elif i == n - 1 and rms_ratio < high_thresh:
            label = "Outro"
        elif rms_ratio >= high_thresh:
            chorus_count += 1
            label = f"Chorus {chorus_count}" if chorus_count > 1 else "Chorus"
        elif rms_ratio < low_thresh and i > 0 and i < n - 1:
            bridge_count += 1
            if bridge_count == 1:
                label = "Bridge"
            else:
                label = "Break"
        else:
            verse_count += 1
            label = f"Verse {verse_count}" if verse_count > 1 else "Verse"

        seg["label"] = label
        
        # Energy description
        if rms_ratio >= high_thresh:
            seg["energy_desc"] = "Năng lượng CAO — full band, đỉnh điểm"
        elif rms_ratio >= low_thresh:
            seg["energy_desc"] = "Năng lượng TRUNG BÌNH — cân bằng"
        else:
            seg["energy_desc"] = "Năng lượng THẤP — nhẹ nhàng, tối giản"

        labeled.append(seg)

    return labeled
