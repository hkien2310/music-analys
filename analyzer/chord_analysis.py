"""
chord_analysis.py — Phân tích vòng hợp âm (Chord Progression).

Xây dựng thư viện 72 chord templates (Major, Minor, Dim, Dom7, Min7, Sus4),
gán chord theo từng beat bằng cosine similarity, smooth sequence,
và tìm pattern lặp lại phổ biến nhất.
"""

import numpy as np
import librosa

from .config import (
    PITCH_CLASSES,
    MAJOR_TEMPLATE, MINOR_TEMPLATE, DIM_TEMPLATE,
    DOM7_TEMPLATE, MIN7_TEMPLATE, SUS4_TEMPLATE,
)
from .utils import cosine_sim, fmt_time, progress

# ─── MODULE 1E: CHORD PROGRESSION ────────────────────────────────────────────

def analyze_chords(y, sr, beat_times, key_info):
    """Phân tích vòng hợp âm theo từng beat"""
    progress("Phân tích hợp âm (Chord Progression)...")

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=24)
    
    # Build chord template library: Major, Minor, Dim, Dom7, Min7, Sus4 = 72 chords
    templates = []
    chord_names = []
    
    for i in range(12):
        # Major (e.g. C, D, E...)
        tmaj = np.roll(MAJOR_TEMPLATE, i)
        templates.append(tmaj / (tmaj.sum() + 1e-9))
        chord_names.append(f"{PITCH_CLASSES[i]}")
        # Minor (e.g. Cm, Dm...)
        tmin = np.roll(MINOR_TEMPLATE, i)
        templates.append(tmin / (tmin.sum() + 1e-9))
        chord_names.append(f"{PITCH_CLASSES[i]}m")
        # Diminished (e.g. Cdim)
        tdim = np.roll(DIM_TEMPLATE, i)
        templates.append(tdim / (tdim.sum() + 1e-9))
        chord_names.append(f"{PITCH_CLASSES[i]}dim")
        # Dominant 7th (e.g. C7)
        tdom7 = np.roll(DOM7_TEMPLATE, i)
        templates.append(tdom7 / (tdom7.sum() + 1e-9))
        chord_names.append(f"{PITCH_CLASSES[i]}7")
        # Minor 7th (e.g. Cm7)
        tmin7 = np.roll(MIN7_TEMPLATE, i)
        templates.append(tmin7 / (tmin7.sum() + 1e-9))
        chord_names.append(f"{PITCH_CLASSES[i]}m7")
        # Sus4 (e.g. Csus4)
        tsus4 = np.roll(SUS4_TEMPLATE, i)
        templates.append(tsus4 / (tsus4.sum() + 1e-9))
        chord_names.append(f"{PITCH_CLASSES[i]}sus4")
    
    templates = np.array(templates)  # shape (72, 12)

    # Gán chord theo từng segment (mỗi beat hoặc nửa beat)
    chord_sequence = []
    hop_length = 512
    sr_frames  = sr / hop_length

    for t_idx, t in enumerate(beat_times):
        # Lấy frame của beat
        frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        frame = max(0, min(frame, chroma.shape[1] - 1))

        # Lấy chroma window ±2 frames để tránh noise
        start_f = max(0, frame - 2)
        end_f   = min(chroma.shape[1], frame + 4)
        chroma_seg = np.mean(chroma[:, start_f:end_f], axis=1)
        chroma_seg = chroma_seg / (chroma_seg.sum() + 1e-9)

        # Cosine similarity với tất cả templates
        sims = np.array([cosine_sim(chroma_seg, tmpl) for tmpl in templates])
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        
        chord_sequence.append({
            "time"      : round(float(t), 2),
            "time_str"  : fmt_time(float(t)),
            "chord"     : chord_names[best_idx],
            "confidence": round(best_sim, 3)
        })

    # Smooth chord sequence - xóa chord đơn lẻ (xuất hiện chỉ 1 beat)
    if len(chord_sequence) > 2:
        smoothed = chord_sequence.copy()
        for i in range(1, len(smoothed) - 1):
            if (smoothed[i]["chord"] != smoothed[i-1]["chord"] and
                    smoothed[i]["chord"] != smoothed[i+1]["chord"]):
                smoothed[i]["chord"] = smoothed[i-1]["chord"]
        chord_sequence = smoothed

    # Tìm chord progression pattern lặp lại
    # Nhóm liên tiếp cùng chord
    compressed = []
    if chord_sequence:
        cur_chord = chord_sequence[0]["chord"]
        cur_start = chord_sequence[0]["time"]
        for cs in chord_sequence[1:]:
            if cs["chord"] != cur_chord:
                compressed.append({
                    "chord": cur_chord,
                    "start": cur_start,
                    "start_str": fmt_time(cur_start)
                })
                cur_chord = cs["chord"]
                cur_start = cs["time"]
        compressed.append({
            "chord": cur_chord,
            "start": cur_start,
            "start_str": fmt_time(cur_start)
        })

    # Tìm vòng hợp âm phổ biến nhất (window 4 chords)
    chord_4grams = {}
    chord_names_only = [c["chord"] for c in compressed]
    for i in range(len(chord_names_only) - 3):
        pattern = " → ".join(chord_names_only[i:i+4])
        chord_4grams[pattern] = chord_4grams.get(pattern, 0) + 1

    # Tìm vòng 3 chords
    chord_3grams = {}
    for i in range(len(chord_names_only) - 2):
        pattern = " → ".join(chord_names_only[i:i+3])
        chord_3grams[pattern] = chord_3grams.get(pattern, 0) + 1

    # Tìm vòng 2 chords
    chord_2grams = {}
    for i in range(len(chord_names_only) - 1):
        pattern = " → ".join(chord_names_only[i:i+2])
        chord_2grams[pattern] = chord_2grams.get(pattern, 0) + 1

    main_progression_4 = max(chord_4grams, key=chord_4grams.get) if chord_4grams else "N/A"
    main_progression_3 = max(chord_3grams, key=chord_3grams.get) if chord_3grams else "N/A"

    # Đếm chord xuất hiện nhiều nhất
    chord_freq = {}
    for c in chord_names_only:
        chord_freq[c] = chord_freq.get(c, 0) + 1
    top_chords = sorted(chord_freq.items(), key=lambda x: x[1], reverse=True)[:8]

    return {
        "sequence"         : compressed,
        "main_prog_4"      : main_progression_4,
        "main_prog_3"      : main_progression_3,
        "top_chords"       : top_chords,
        "unique_chords"    : len(chord_freq),
        "full_beat_seq"    : chord_sequence[:50],  # giới hạn 50 beats để output
    }
