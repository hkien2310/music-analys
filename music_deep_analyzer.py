#!/usr/bin/env python3
"""
music_deep_analyzer.py
======================
Phân tích bài nhạc cực kỳ chi tiết → báo cáo Markdown + Suno AI Prompt

Cách dùng:
    python3 music_deep_analyzer.py <file_âm_thanh.mp3>

Output:
    <tên_bài>_analysis.md    — Báo cáo nhạc lý đầy đủ
    <tên_bài>_suno_prompt.txt — Prompt tối ưu cho Suno AI
"""

import os
import sys
import json
import time
import warnings
import datetime
warnings.filterwarnings("ignore")

# ─── IMPORT & DEPENDENCY DETECTION ────────────────────────────────────────────

DEPS = {}

# Bắt buộc
try:
    import numpy as np
    import librosa
    import librosa.display
    DEPS["librosa"] = True
    DEPS["numpy"] = True
except ImportError as e:
    print(f"[LỖI NGHIÊM TRỌNG] Thiếu librosa/numpy: {e}")
    print("Cài đặt: pip install librosa numpy")
    sys.exit(1)

# Metadata
try:
    from tinytag import TinyTag
    DEPS["tinytag"] = True
except ImportError:
    DEPS["tinytag"] = False

# Loudness chuẩn LUFS
try:
    import pyloudnorm as pyln
    DEPS["pyloudnorm"] = True
except ImportError:
    DEPS["pyloudnorm"] = False

# madmom - chord detection chính xác nhất
try:
    import madmom
    from madmom.features.chords import DeepChromaChordRecognitionProcessor
    from madmom.audio.chroma import DeepChromaProcessor
    DEPS["madmom"] = True
except Exception:
    DEPS["madmom"] = False

# scipy - signal processing
try:
    from scipy.signal import medfilt
    from scipy.ndimage import median_filter
    DEPS["scipy"] = True
except ImportError:
    DEPS["scipy"] = False

# ─── CẤU HÌNH ─────────────────────────────────────────────────────────────────

PITCH_CLASSES    = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ENHARMONIC       = {'C#':'Db','D#':'Eb','F#':'Gb','G#':'Ab','A#':'Bb'}

# Chord triad templates (3 nốt thực sự)
# Major:  root + major 3rd (4 semitones) + perfect 5th (7 semitones)
# Minor:  root + minor 3rd (3 semitones) + perfect 5th (7 semitones)
# Dim:    root + minor 3rd (3) + diminished 5th (6)
# Dom7:   root + major 3rd (4) + 5th (7) + minor 7th (10)
# Min7:   root + minor 3rd (3) + 5th (7) + minor 7th (10)
MAJOR_TEMPLATE   = np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float)  # [0,4,7]
MINOR_TEMPLATE   = np.array([1,0,0,1,0,0,0,1,0,0,0,0], dtype=float)  # [0,3,7]
DIM_TEMPLATE     = np.array([1,0,0,1,0,0,1,0,0,0,0,0], dtype=float)  # [0,3,6]
DOM7_TEMPLATE    = np.array([1,0,0,0,1,0,0,1,0,0,1,0], dtype=float)  # [0,4,7,10]
MIN7_TEMPLATE    = np.array([1,0,0,1,0,0,0,1,0,0,1,0], dtype=float)  # [0,3,7,10]
SUS4_TEMPLATE    = np.array([1,0,0,0,0,1,0,1,0,0,0,0], dtype=float)  # [0,5,7]

# Profile Krumhansl-Schmuckler để detect key
KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

# Mapping genre từ metadata sang Suno tags
GENRE_MAP = {
    "pop": ["pop", "contemporary"],
    "rock": ["rock", "electric guitar", "distortion"],
    "metal": ["metal", "heavy guitar", "double kick drums"],
    "jazz": ["jazz", "swing", "improvisation", "saxophone"],
    "classical": ["orchestral", "classical", "strings ensemble"],
    "electronic": ["electronic", "synthesizer", "EDM"],
    "hip hop": ["hip-hop", "rap", "808 bass", "trap"],
    "hip-hop": ["hip-hop", "rap", "808 bass"],
    "r&b": ["R&B", "soul", "smooth", "groove"],
    "rnb": ["R&B", "soul", "groove"],
    "country": ["country", "acoustic guitar", "twang"],
    "folk": ["folk", "acoustic", "storytelling"],
    "reggae": ["reggae", "offbeat guitar", "bass-heavy"],
    "blues": ["blues", "12-bar", "slide guitar"],
    "soul": ["soul", "gospel", "soulful vocals"],
    "latin": ["latin", "percussion", "rhythmic"],
    "k-pop": ["K-pop", "polished production", "catchy hook"],
    "kpop": ["K-pop", "polished production"],
    "indie": ["indie", "lo-fi", "bedroom production"],
    "alternative": ["alternative", "indie", "unconventional"],
    "punk": ["punk", "fast", "raw energy", "power chords"],
    "dance": ["dance", "four-on-the-floor", "club"],
    "house": ["house", "four-on-the-floor", "synth bass"],
    "techno": ["techno", "industrial", "repetitive beat"],
    "trap": ["trap", "hi-hats", "808 bass", "dark"],
    "lo-fi": ["lo-fi", "vinyl crackle", "warm", "chill"],
}

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

# ─── MODULE 1A: METADATA ──────────────────────────────────────────────────────

def get_metadata(file_path):
    meta = {
        "title": None, "artist": None, "album": None,
        "year": None, "genre": None, "track": None, "comment": None
    }
    if DEPS["tinytag"]:
        try:
            tag = TinyTag.get(file_path)
            meta["title"]   = tag.title
            meta["artist"]  = tag.artist
            meta["album"]   = tag.album
            meta["year"]    = tag.year
            meta["genre"]   = tag.genre
            meta["track"]   = tag.track
            meta["comment"] = tag.comment
        except Exception:
            pass
    # Fallback: tên file
    if not meta["title"]:
        meta["title"] = os.path.splitext(os.path.basename(file_path))[0]
    return meta

# ─── MODULE 1B: AUDIO LOADING ─────────────────────────────────────────────────

def load_audio(file_path):
    """Load audio, trả về (y_mono, y_stereo_or_mono, sr, duration)"""
    progress("Đang tải file âm thanh...")
    # Giữ sample rate gốc
    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    # Tải stereo để check channels
    try:
        y_orig, _ = librosa.load(file_path, sr=None, mono=False)
        channels = y_orig.ndim
    except Exception:
        channels = 1
    return y, sr, duration, channels

# ─── MODULE 1B2: SOURCE SEPARATION (DEMUCS) ───────────────────────────────────

def separate_stems(audio_path, out_dir="separated"):
    """Tách âm thanh thành stems bằng Demucs (2 stems: vocals và no_vocals)"""
    progress("Đang tách âm thanh bằng Demucs (có thể mất vài phút)...")
    
    import subprocess
    import os
    import shutil
    
    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run([
            "demucs", "-n", "htdemucs", "--two-stems", "vocals", 
            audio_path, "-o", out_dir
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        # Demucs output structure: out_dir/htdemucs/<base_name>/vocals.wav
        vocals_path = os.path.join(out_dir, "htdemucs", base_name, "vocals.wav")
        no_vocals_path = os.path.join(out_dir, "htdemucs", base_name, "no_vocals.wav")
        
        if os.path.exists(vocals_path) and os.path.exists(no_vocals_path):
            return vocals_path, no_vocals_path
        else:
            progress("Không tìm thấy file tách sau khi chạy Demucs.")
            return None, None
    except Exception as e:
        progress(f"Lỗi khi chạy Demucs: {e}")
        return None, None

def cleanup_stems(out_dir="separated"):
    import shutil
    try:
        shutil.rmtree(out_dir, ignore_errors=True)
    except:
        pass

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
    result["dynamic_range_db"] = round(float(20 * np.log10(result["rms_max"] / (result["rms_min"] + 1e-9))), 1)

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

# ─── MODULE 1F: LYRICS TRANSCRIPTION ──────────────────────────────────────────

def analyze_lyrics(audio_path):
    """Sử dụng OpenAI Whisper để bóc băng lời bài hát."""
    progress("Đang bóc băng lời bài hát bằng Whisper (có thể mất thời gian)...")
    try:
        import whisper
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return result.get("text", "").strip()
    except ImportError:
        return "(Whisper chưa được cài đặt)"
    except Exception as e:
        progress(f"Lỗi khi bóc băng Whisper: {e}")
        return "(Lỗi khi bóc băng)"

# ─── MODULE 1G: INSTRUMENTS & AUDIO TAGS (PANNs) ──────────────────────────────

def analyze_instruments(audio_path):
    """Sử dụng PANNs (AudioSet) để nhận diện nhạc cụ, giọng hát và không gian."""
    progress("Phân tích nhạc cụ & âm thanh bằng PANNs (Audio Tagging)...")
    try:
        from panns_inference import AudioTagging, labels
        import librosa
        import numpy as np
        import warnings

        # PANNs yêu cầu sample rate 32kHz
        audio, _ = librosa.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, samples)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Tự động tải model Cnn14 (chạy trên CPU)
            at = AudioTagging(checkpoint_path=None, device='cpu')
            clipwise_output, _ = at.inference(audio)

        sorted_indexes = np.argsort(clipwise_output[0])[::-1]
        
        tags = []
        # Các nhãn quá chung chung, không hữu ích cho Suno prompt
        ignore_tags = {
            "Music", "Musical instrument", "Sound", "Song", "Noise",
            "Inside, small room", "Inside, large room or hall", "Speech",
            "Music of Africa", "Music of Asia", "Music of Latin America"
        }

        for i in range(20):
            idx = sorted_indexes[i]
            label = np.array(labels)[idx]
            prob = float(clipwise_output[0][idx])
            if label not in ignore_tags and prob >= 0.01:
                tags.append(label)

        return tags[:6]  # Trả về tối đa 6 nhãn phù hợp nhất
    except ImportError:
        return ["(panns-inference chưa được cài đặt)"]
    except Exception as e:
        progress(f"Lỗi khi chạy PANNs: {e}")
        return []

# ─── MODULE 2: GENERATE ANALYSIS REPORT ──────────────────────────────────────

def build_report(file_path, meta, audio_props, rhythm, key_info, chords,
                 structure, dynamics, timbre, extra, audio_tags, lyrics=""):
    """
    Tạo báo cáo Markdown tổng hợp tất cả kết quả phân tích.
    """
    progress("Đang tạo báo cáo phân tích Markdown...")

    lines = []
    A = lines.append  # shortcut

    song_name = meta.get("title") or os.path.splitext(os.path.basename(file_path))[0]

    # ── HEADER ──
    A(f"# 🎵 Báo cáo Phân tích Nhạc lý: {song_name}")
    A(f"> Tạo bởi **music_deep_analyzer.py** lúc {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"> File: `{os.path.basename(file_path)}`")
    A("")

    # ── TÓM TẮT NHANH ──
    A("## ⚡ Tóm tắt nhanh")
    A("")
    A("| Thông số | Giá trị |")
    A("|----------|---------|")
    A(f"| 🎹 Tông nhạc | **{key_info['full']}** (confidence: {key_info['confidence_pct']}%) |")
    A(f"| 🥁 Tempo | **{rhythm['tempo_bpm']} BPM** |")
    A(f"| ⏱️ Nhịp phách | **{rhythm['time_signature']}** |")
    A(f"| ⏳ Thời lượng | **{audio_props['duration_str']}** ({audio_props['duration_sec']}s) |")
    A(f"| 🔊 Loudness | **{dynamics.get('loudness_lufs', 'N/A')} LUFS** |")
    A(f"| 💃 Danceability | **{extra['danceability_score']}** — {extra['danceability_label']} |")
    A(f"| 🎭 Valence/Mood | **{extra['valence_score']}** — {extra['valence_label']} |")
    A(f"| 🎸 Phong cách | {timbre['hp_description']} |")
    A(f"| 🌈 Âm sắc | {timbre['brightness']} |")
    A(f"| 🎧 Âm thanh (PANNs)| **{', '.join(audio_tags)}** |")
    A("")

    # ── METADATA ──
    A("---")
    A("## 📋 1. Thông tin Metadata")
    A("")
    A(f"- **Tên bài:** {meta.get('title') or '(không có)'}")
    A(f"- **Nghệ sĩ:** {meta.get('artist') or '(không có)'}")
    A(f"- **Album:** {meta.get('album') or '(không có)'}")
    A(f"- **Năm:** {meta.get('year') or '(không có)'}")
    A(f"- **Thể loại:** {meta.get('genre') or '(không có)'}")
    A(f"- **Track số:** {meta.get('track') or '(không có)'}")
    A("")

    # ── AUDIO PROPERTIES ──
    A("---")
    A("## 🔧 2. Thuộc tính Kỹ thuật Âm thanh")
    A("")
    A(f"- **Thời lượng:** {audio_props['duration_str']} ({audio_props['duration_sec']} giây)")
    A(f"- **Sample Rate:** {audio_props['sample_rate']} Hz")
    A(f"- **Tổng số mẫu:** {audio_props['total_samples']:,}")
    A(f"- **Channels:** {audio_props['channels']}")
    A(f"- **Kích thước file:** {audio_props['file_size_mb']} MB")
    A("")

    # ── RHYTHM ──
    A("---")
    A("## 🥁 3. Phân tích Nhịp điệu")
    A("")
    A(f"### Tempo: **{rhythm['tempo_bpm']} BPM**")
    A("")
    A(f"- **Nhịp phách (Time Signature):** {rhythm['time_signature']} _(confidence: {rhythm['time_sig_confidence']})_")
    A(f"- **Tổng số beats phát hiện:** {rhythm['beats_total']}")
    A(f"- **Độ ổn định nhịp (Beat Regularity):** {rhythm['tempo_stability']} / 1.0")
    A(f"- **Mật độ events (Onset Density):** {rhythm['onset_density_per_sec']} events/giây")
    A("")
    
    # Diễn giải tempo
    bpm = rhythm["tempo_bpm"]
    if bpm < 60:
        tempo_feel = "Rất chậm (Grave/Largo) — nhạc thư giãn, điện ảnh"
    elif bpm < 80:
        tempo_feel = "Chậm (Andante) — ballad, acoustic"
    elif bpm < 100:
        tempo_feel = "Vừa phải (Moderato) — pop nhẹ nhàng"
    elif bpm < 120:
        tempo_feel = "Sôi nổi (Allegretto) — pop, rock nhẹ"
    elif bpm < 140:
        tempo_feel = "Nhanh (Allegro) — dance pop, rock"
    elif bpm < 160:
        tempo_feel = "Rất nhanh (Vivace) — EDM, punk, metal"
    else:
        tempo_feel = "Cực nhanh (Presto) — drum & bass, thrash"
    
    A(f"**Cảm giác nhịp:** {tempo_feel}")
    A("")

    # ── KEY ──
    A("---")
    A("## 🎼 4. Phân tích Tông nhạc (Key & Scale)")
    A("")
    A(f"### Tông nhạc chính: **{key_info['full']}** (confidence: {key_info['confidence_pct']}%)")
    A("")
    A(f"- **Nốt gốc (Root):** {key_info['root']}")
    A(f"- **Thang âm (Scale):** {key_info['scale']}")
    A(f"- **Tông song song:** {key_info['relative_key']}")
    A("")
    A("**Top 5 ứng viên tông nhạc:**")
    A("")
    A("| Hạng | Tông nhạc | Điểm tương quan |")
    A("|------|-----------|-----------------|")
    for rank, (note, scale, score) in enumerate(key_info['top_candidates'], 1):
        marker = "← **Được chọn**" if rank == 1 else ""
        A(f"| {rank} | {note} {scale} | {score} {marker}|")
    A("")
    
    # Nốt trong thang âm
    root_idx = PITCH_CLASSES.index(key_info['root'])
    if key_info['scale'] == "Major":
        intervals = [0, 2, 4, 5, 7, 9, 11]
        degree_names = ["I (Chủ âm)", "II", "III", "IV (Hạ át)", "V (Át âm)", "VI", "VII"]
    else:
        intervals = [0, 2, 3, 5, 7, 8, 10]
        degree_names = ["i (Chủ âm)", "ii°", "III", "iv", "v/V", "VI", "VII"]
    
    A("**Các nốt trong thang âm này:**")
    A("")
    scale_notes = [PITCH_CLASSES[(root_idx + iv) % 12] for iv in intervals]
    A(f"> {' — '.join(scale_notes)}")
    A("")
    
    A("**Hợp âm diatonic (nằm trong tông) phổ biến:**")
    A("")
    if key_info['scale'] == "Major":
        chord_notes = [
            (scale_notes[0], ""), (scale_notes[1], "m"), (scale_notes[2], "m"),
            (scale_notes[3], ""), (scale_notes[4], ""), (scale_notes[5], "m"),
            (scale_notes[6], "dim")
        ]
    else:
        chord_notes = [
            (scale_notes[0], "m"), (scale_notes[1], "dim"), (scale_notes[2], ""),
            (scale_notes[3], "m"), (scale_notes[4], "m"), (scale_notes[5], ""),
            (scale_notes[6], "")
        ]
    diatonic_str = " — ".join([f"{n}{q}" for n, q in chord_notes])
    A(f"> {diatonic_str}")
    A("")

    # ── CHORD PROGRESSION ──
    A("---")
    A("## 🎸 5. Phân tích Hợp âm (Chord Progression)")
    A("")
    A(f"**Vòng hợp âm chủ đạo (4 hợp âm):** `{chords['main_prog_4']}`")
    A(f"**Vòng hợp âm chủ đạo (3 hợp âm):** `{chords['main_prog_3']}`")
    A(f"**Số hợp âm khác nhau:** {chords['unique_chords']}")
    A("")
    A("**Top hợp âm xuất hiện nhiều nhất:**")
    A("")
    A("| Hợp âm | Số lần xuất hiện |")
    A("|--------|-----------------|")
    for chord_name, count in chords['top_chords']:
        A(f"| **{chord_name}** | {count} lần |")
    A("")
    
    A("**Diễn biến hợp âm theo thời gian (tóm tắt):**")
    A("")
    A("| Thời gian | Hợp âm |")
    A("|-----------|--------|")
    MAX_CHORD_ROWS = 40
    seq = chords["sequence"]
    step = max(1, len(seq) // MAX_CHORD_ROWS)
    for i, item in enumerate(seq):
        if i % step == 0:
            A(f"| {item['start_str']} | **{item['chord']}** |")
    A("")

    # ── SONG STRUCTURE ──
    A("---")
    A("## 🏗️ 6. Cấu trúc Bài nhạc")
    A("")
    A("| # | Đoạn | Thời gian | Thời lượng | Năng lượng | Mô tả |")
    A("|---|------|-----------|------------|------------|-------|")
    for seg in structure:
        energy_bar = ascii_bar(seg["rms_ratio"], 1.0, width=10)
        A(f"| {seg['index']} | **{seg['label']}** | {seg['start_str']}–{seg['end_str']} | {seg['duration']}s | `{energy_bar}` | {seg['energy_desc']} |")
    A("")
    
    # Chi tiết từng đoạn
    A("### Chi tiết từng đoạn:")
    A("")
    for seg in structure:
        A(f"#### {seg['label']} `[{seg['start_str']} → {seg['end_str']}]`")
        A(f"- Thời lượng: **{seg['duration']} giây**")
        A(f"- Năng lượng trung bình: {seg['avg_rms']} RMS (tỷ lệ: {seg['rms_ratio']})")
        A(f"- Spectral Centroid: {seg['spec_cent']} Hz")
        A(f"- Trạng thái: {seg['energy_desc']}")
        A("")

    # ── DYNAMICS ──
    A("---")
    A("## 📈 7. Dynamics & Loudness")
    A("")
    A(f"- **RMS trung bình:** {dynamics['rms_avg']}")
    A(f"- **RMS cực đại:** {dynamics['rms_max']}")
    A(f"- **Dynamic Range:** {dynamics['dynamic_range_db']} dB")
    if dynamics.get('loudness_lufs'):
        A(f"- **Loudness (LUFS):** {dynamics['loudness_lufs']} LUFS")
        A(f"- **Đánh giá:** {dynamics.get('loudness_note', '')}")
    A("")
    
    # ASCII energy timeline
    A("### Biểu đồ năng lượng theo thời gian:")
    A("")
    A("```")
    timeline = dynamics["energy_timeline"]
    max_rms = max(t["rms"] for t in timeline) if timeline else 0.001
    for item in timeline:
        bar = ascii_bar(item["rms"], max_rms, width=30)
        A(f"{fmt_time(item['time']):>5}  {bar}  {item['rms']:.4f}")
    A("```")
    A("")

    # ── TIMBRE ──
    A("---")
    A("## 🎨 8. Phân tích Âm sắc (Timbre & Spectral)")
    A("")
    A(f"- **Âm sắc tổng thể:** {timbre['brightness']}")
    A(f"- **Chất lượng âm:** {timbre['tonal_quality']}")
    A(f"- **Phong cách:** {timbre['hp_description']}")
    A("")
    A("### Đặc trưng phổ:")
    A("")
    A(f"- **Spectral Centroid:** {timbre['spectral_centroid_hz']} Hz — điểm 'trọng tâm' tần số")
    A(f"- **Spectral Bandwidth:** {timbre['spectral_bandwidth_hz']} Hz — độ rộng dải phổ")
    A(f"- **Spectral Rolloff:** {timbre['spectral_rolloff_hz']} Hz — điểm 85% năng lượng phổ tập trung dưới đây")
    A(f"- **Spectral Flatness:** {timbre['spectral_flatness']} — 0=thuần tonal, 1=thuần noise")
    A(f"- **Zero-Crossing Rate:** {timbre['zero_crossing_rate']} — độ biến đổi tín hiệu")
    A("")
    A("### Harmonic vs Percussive:")
    A("")
    harm_bar = ascii_bar(timbre['harmonic_ratio'], 1.0, width=20)
    perc_bar = ascii_bar(timbre['percussive_ratio'], 1.0, width=20)
    A(f"- **Harmonic (giai điệu):** `{harm_bar}` {round(timbre['harmonic_ratio']*100)}%")
    A(f"- **Percussive (bộ gõ):** `{perc_bar}` {round(timbre['percussive_ratio']*100)}%")
    A("")
    A(f"**Các nốt nhạc chiếm ưu thế:** {', '.join(timbre['dominant_notes'])}")
    A("")

    # ── EXTRA ──
    A("---")
    A("## 🔬 9. Đặc trưng Cảm xúc & Nhạc cụ")
    A("")
    A("### Cảm xúc & Chuyển động:")
    A(f"- **Danceability:** {extra['danceability_score']} / 1.0 — {extra['danceability_label']}")
    A(f"- **Valence (tâm trạng):** {extra['valence_score']} / 1.0 — {extra['valence_label']}")
    A("")
    A("### Nhận diện Âm thanh / Nhạc cụ (Audio Tags):")
    A(f"> {', '.join(audio_tags)}")
    A("")

    # ── FOOTER ──
    # Lời bài hát
    A("---")
    A("## 🎤 10. Lời bài hát (Whisper)")
    A("")
    if lyrics and lyrics.strip():
        A("```text")
        A(lyrics.strip())
        A("```")
    else:
        A("*Không tìm thấy lời bài hát hoặc file không có vocal.*")
    A("")
    A("---")
    A("## 📌 Ghi chú về độ chính xác")
    A("")
    A("> **Tông nhạc:** Độ chính xác ~70–85% (heuristic, tốt nhất với nhạc phương Tây)")
    A("> **Hợp âm:** Độ chính xác ~65–75% với nhạc rõ ràng, thấp hơn với nhạc heavy distortion")
    A("> **Cấu trúc:** Phân đoạn tự động dựa trên năng lượng, label có thể không hoàn toàn chính xác")
    A("> **Danceability/Valence:** Ước tính heuristic, không phải mô hình ML")
    A("")
    A("*Được tạo bởi music_deep_analyzer.py — librosa + pyloudnorm + tinytag*")

    return "\n".join(lines)


# ─── MODULE 3: GENERATE SUNO PROMPT ──────────────────────────────────────────

def build_suno_prompt(meta, rhythm, key_info, chords, structure, dynamics, timbre, extra, audio_tags, lyrics=""):
    """
    Tạo ra một file văn bản chứa prompt copy-paste trực tiếp cho Suno AI.
    """
    progress("Dang tao Suno AI Prompt...")

    bpm        = rhythm["tempo_bpm"]
    ts         = rhythm.get("time_signature", "4/4")
    key_full   = key_info["full"]
    root       = key_info["root"]
    scale      = key_info["scale"]
    valence    = extra["valence_score"]
    dance      = extra["danceability_score"]
    sc_hz      = timbre["spectral_centroid_hz"]
    harm_ratio = timbre["harmonic_ratio"]
    perc_ratio = timbre["percussive_ratio"]
    beat_reg   = rhythm.get("tempo_stability", 0.8)
    genre_raw  = (meta.get("genre") or "").lower().strip()

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: DETECT SONIC ARCHETYPE
    # The archetype shapes the ENTIRE prompt — vocabulary, adjectives, feel
    # ────────────────────────────────────────────────────────────────────────
    def _detect_archetype():
        # Keyword shortcuts from metadata genre
        kw_map = {
            "hip hop": "hiphop", "hip-hop": "hiphop", "rap": "hiphop",
            "r&b": "rnb", "soul": "gospel-soul", "gospel": "gospel-soul",
            "jazz": "jazz", "blues": "blues",
            "metal": "metal", "heavy": "metal",
            "ambient": "ambient", "classical": "ambient",
            "electronic": "edm", "edm": "edm", "techno": "edm", "house": "edm",
            "folk": "folk-rock", "country": "folk-rock", "bluegrass": "folk-rock",
            "rock": "alt-rock", "punk": "alt-rock", "grunge": "alt-rock",
            "indie": "indie-pop",
        }
        for kw, arch in kw_map.items():
            if kw in genre_raw: return arch

        # Feature-based detection (ordered by specificity)
        if bpm > 135 and perc_ratio > 0.40:
            return "edm"
        if bpm > 120 and perc_ratio > 0.30 and sc_hz > 3500:
            return "upbeat-pop"
        if valence > 0.65 and dance > 0.70 and scale == "Major":
            return "gospel-soul"
        if scale == "Minor" and valence < 0.35 and bpm < 110:
            return "dark-alt"
        if harm_ratio > 0.85 and bpm < 85:
            return "folk-rock" if sc_hz > 1800 else "ambient"
        if harm_ratio > 0.75 and bpm < 100:
            return "folk-rock"
        if sc_hz < 1100 and harm_ratio > 0.65:
            return "rnb"
        if bpm >= 95 and sc_hz > 2500 and valence > 0.50:
            return "indie-pop"
        if scale == "Minor" and valence < 0.50:
            return "alt-rock"
        return "contemporary-pop"

    archetype = _detect_archetype()

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: GENRE + INFLUENCES (evocative, specific to archetype)
    # ────────────────────────────────────────────────────────────────────────
    GENRE_STRINGS = {
        "gospel-soul":      "Gospel-soul with indie-folk warmth",
        "folk-rock":        "Indie folk-rock with Americana and roots-rock influences",
        "alt-rock":         "Brooding alternative rock with post-punk undertones",
        "dark-alt":         "Dark alternative rock with gothic and shoegaze undertones",
        "indie-pop":        "Sunlit indie pop with bedroom-pop and dream-pop influences",
        "contemporary-pop": "Contemporary pop with alternative and soul influences",
        "upbeat-pop":       "Contemporary pop with funk and gospel energy",
        "rnb":              "Smooth neo-soul with classic R&B and jazz influences",
        "edm":              "Electronic dance music with progressive house influences",
        "hiphop":           "Hip-hop with boom-bap and soul sample influences",
        "ambient":          "Cinematic ambient with post-rock and orchestral influences",
        "jazz":             "Jazz-influenced indie with sophisticated harmonic textures",
        "blues":            "Blues-rock with Southern soul and gospel influences",
        "metal":            "Heavy metal with progressive and alternative influences",
    }

    # If scale is Minor, skew toward darker vocabulary
    genre_str = GENRE_STRINGS.get(archetype, "Contemporary pop with alternative influences")
    if scale == "Minor" and "soul" in genre_str:
        genre_str = genre_str.replace("warmth", "melancholy")
    if scale == "Minor" and archetype == "indie-pop":
        genre_str = "Melancholic indie pop with dream-pop and shoegaze influences"

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: LEAD RHYTHM INSTRUMENT (specific technique + sonic character)
    # ────────────────────────────────────────────────────────────────────────
    def _lead_instrument():
        # Guitar articulation based on context
        if archetype in ("folk-rock", "gospel-soul", "blues"):
            if bpm < 90:
                return "a prominent acoustic guitar fingerpicking intricate melodic arpeggios"
            if harm_ratio > 0.80:
                return "a prominent acoustic guitar strumming driving eighth-note patterns"
            return "an acoustic guitar strumming syncopated chord patterns with percussive pull-offs"

        if archetype in ("alt-rock", "dark-alt", "metal"):
            if valence < 0.30:
                return "a distorted electric guitar playing angular, stabbing power chord riffs"
            if bpm > 110:
                return "a wiry electric guitar delivering tight, palm-muted riff patterns"
            return "a brooding electric guitar with heavy chord voicings and subtle feedback"

        if archetype == "indie-pop":
            if sc_hz > 3200:
                return "a chiming electric guitar with clean, reverb-washed arpeggios"
            return "a jangly electric guitar with bright, open chord strumming"

        if archetype in ("contemporary-pop", "upbeat-pop"):
            if harm_ratio > 0.75 and bpm < 115:
                return "an acoustic guitar strumming crisp driving eighth-note patterns"
            if bpm > 115:
                return "a bright electric guitar with crisp, strummed eighth notes"
            return "a clean electric guitar playing rhythmic syncopated chord stabs"

        if archetype == "rnb":
            return "a smooth electric guitar with clean, chord-melody fills and subtle embellishments"

        if archetype == "edm":
            return "driving synthesizer arpeggios and pulsing sawtooth stabs locked to the grid"

        if archetype in ("ambient", "jazz"):
            if harm_ratio > 0.85:
                return "a fingerpicked acoustic guitar weaving delicate melodic counterpoint"
            return "a clean electric guitar with spacious, effects-laden chord voicings"

        return "a prominent acoustic guitar strumming driving eighth-note patterns"

    # ────────────────────────────────────────────────────────────────────────
    # STEP 4: BASS DESCRIPTION (groove character + playing style)
    # ────────────────────────────────────────────────────────────────────────
    def _bass_desc():
        if archetype == "edm":
            return "a crushing mechanical sub-bass providing deep low-end pressure"
        if archetype == "metal":
            return "a distorted bass guitar doubling the guitar riffs with heavy attack"
        if archetype == "rnb":
            return "a melodic electric bass with slap-and-pop accents and deep groove"
        if archetype == "jazz":
            return "a walking acoustic upright bass providing harmonic and rhythmic momentum"
        if archetype == "ambient":
            return "a sparse bass guitar providing subtle low-end support"
        if dance > 0.75 and bpm > 105:
            return "a syncopated electric bass driving the rhythmic groove with locked-in precision"
        if bpm < 80:
            return "a sparse, melodic bass guitar anchoring the harmonic foundation"
        if beat_reg > 0.88:
            return "a steady electric bass providing a deep, pulsing rhythmic foundation"
        return "a syncopated electric bass weaving around the beat with fluid phrasing"

    # ────────────────────────────────────────────────────────────────────────
    # STEP 5: DRUMS (sonic character + specific kit description)
    # ────────────────────────────────────────────────────────────────────────
    def _drum_desc():
        if perc_ratio < 0.05:
            return None
        if archetype == "edm":
            return "a heavy distorted kick drum and sterile, clockwork hi-hat pattern running in rigid 4/4"
        if archetype == "metal":
            return "a thunderous double kick, cracking snare, and blast-beat capable kit"
        if archetype == "ambient":
            if perc_ratio < 0.10:
                return None
            return "minimal brushed percussion with soft, understated kick hits"
        if archetype in ("folk-rock", "gospel-soul", "blues"):
            snare = "heavy, dry snare" if bpm > 100 else "crisp backbeat snare"
            return "a live drum kit with a tight kick, {}, and active hi-hats locking in the groove".format(snare)
        if archetype in ("alt-rock", "dark-alt"):
            return "a pounding drum kit with explosive tom fills and a thunderous, room-shaking snare"
        if archetype == "rnb":
            return "a tight drum machine groove with punchy, compressed snare hits and shuffled hi-hats"
        if archetype == "indie-pop":
            if perc_ratio < 0.15:
                return "subtle percussion with brushed snare and soft kick drum"
            return "a live drum kit with a bouncy, open feel and ringing snare"
        if dance > 0.75 and beat_reg > 0.88:
            snare = "heavy, dry snare" if bpm > 100 else "crisp backbeat snare"
            return "a live drum kit with a tight kick, {}, and active hi-hats".format(snare)
        if perc_ratio < 0.15:
            return "subtle percussion with brushed snare and soft kick drum"
        return "a live drum kit with a solid backbeat and driving hi-hat momentum"

    # ────────────────────────────────────────────────────────────────────────
    # STEP 6: TEXTURE / HARMONIC LAYER (specific instrument + dual function)
    # ────────────────────────────────────────────────────────────────────────
    def _texture_desc():
        if archetype in ("gospel-soul", "blues", "folk-rock") and valence > 0.45:
            return "A Hammond B3 organ provides sustained chordal textures and rhythmic stabs"
        if archetype == "rnb":
            return "A Rhodes electric piano provides warm, shimmering chordal fills and melodic counter-lines"
        if archetype == "jazz":
            return "Sparse piano voicings and brushed vibraphone provide spacious harmonic color"
        if archetype == "ambient":
            return "Expansive reverb-soaked synthesizer pads create an immersive, weightless atmosphere"
        if archetype == "dark-alt":
            return "Atmospheric synthesizer textures dripping with reverb cast a cold, ominous shadow"
        if archetype == "edm":
            return "Lush synthesizer pads swell and pulse in sync with the kick drum sidechain"
        if archetype in ("contemporary-pop", "upbeat-pop") and bpm > 115:
            return "Lush synth pads add shimmer and harmonic depth between the vocal phrases"
        if archetype == "indie-pop":
            if sc_hz > 2800:
                return "Shimmering guitar reverb and subtle synthesizer textures fill the sonic space"
            return "Warm piano chords provide gentle harmonic support and melodic coloring"
        if harm_ratio > 0.80 and sc_hz < 1400:
            return "A piano provides rich, resonant chordal accompaniment anchoring the harmonic center"
        if harm_ratio > 0.75:
            return "Warm string pads swell beneath the arrangement providing lush harmonic depth"
        return None

    # ────────────────────────────────────────────────────────────────────────
    # STEP 7: VOCALS (gender cue + specific quality + ensemble)
    # ────────────────────────────────────────────────────────────────────────
    def _vocal_desc():
        # Lead vocal quality based on archetype + valence
        LEAD_QUALITIES = {
            "gospel-soul": (
                "Lead male vocals are soulful and gritty, with gospel-trained power and conviction"
                if valence < 0.65 else
                "Lead vocals are soaring and euphoric, climbing to roof-raising heights"
            ),
            "folk-rock": (
                "Lead male vocals are warm and weathered, telling stories with quiet intensity"
                if valence < 0.55 else
                "Lead vocals are earnest and heartfelt, with a rugged, road-worn timbre"
            ),
            "alt-rock": (
                "Lead vocals are raw and emotionally unguarded, shifting from whisper to howl"
                if valence < 0.40 else
                "Lead vocals are gritty and expressive, riding the tension between melody and noise"
            ),
            "dark-alt": "Lead vocals are hushed and spectral, delivered with cold, emotionless precision",
            "indie-pop": (
                "Lead female vocals are airy and intimate, with a breathy, confessional quality"
                if valence < 0.55 else
                "Lead vocals are bright and melodically inventive, floating above the mix"
            ),
            "contemporary-pop": (
                "Lead vocals are warm and expressive, carrying the emotional weight with conviction"
                if valence < 0.65 else
                "Lead vocals are powerful and soaring, anthemic and built for arenas"
            ),
            "upbeat-pop": "Lead vocals are exuberant and infectious, dripping with charisma",
            "rnb": "Lead vocals are silky and melismatic, deploying precise runs and intimate ad-libs",
            "edm": "A dry, processed vocal chop provides rhythmic texture between drops",
            "ambient": "Wordless, heavily reverbed vocal harmonics drift through the arrangement like mist",
            "hiphop": "A confident, rhythmically precise MC rides the beat with cadence and flow",
            "jazz": "Lead vocals are smoky and nuanced, phrasing behind the beat with languid ease",
            "blues": "Lead vocals are raw and pleading, bending notes with aching, blue-note expressivity",
            "metal": "Aggressive, throat-shredding vocals alternate between guttural growls and melodic cleans",
        }
        lead = LEAD_QUALITIES.get(archetype, "Lead vocals are warm and expressive, delivering the melody with conviction")

        # Backing vocals based on archetype
        if archetype in ("gospel-soul",) or (valence > 0.60 and dance > 0.65):
            backing = "supported by a large gospel-style choir providing call-and-response and harmonized backing"
        elif archetype in ("folk-rock", "blues"):
            backing = "backed by warm three-part harmonies evoking communal, campfire intimacy"
        elif archetype in ("indie-pop", "contemporary-pop") and valence > 0.50:
            backing = "supported by layered vocal harmonies providing lush, stacked depth"
        elif archetype in ("alt-rock", "dark-alt"):
            backing = "with sparse, dissonant backing harmonies adding unease and tension"
        elif archetype == "rnb":
            backing = "with breathy, interlocking backing vocals weaving around the lead"
        else:
            backing = None

        if backing:
            return "{}, {}".format(lead, backing)
        return lead

    # ────────────────────────────────────────────────────────────────────────
    # STEP 8: RHYTHMIC ACCENT / SPECIAL DETAIL (unique production touch)
    # ────────────────────────────────────────────────────────────────────────
    def _accent_desc():
        if archetype == "edm":
            return "A single screeching synthesizer lead slices through the breakdown before the drop"
        if archetype in ("gospel-soul", "folk-rock", "contemporary-pop") and dance > 0.70:
            return "Handclaps accentuate the backbeat in the choruses, driving communal energy"
        if archetype == "dark-alt":
            return "Reverb-soaked single-coil guitar feedback drones between sections"
        if archetype == "metal":
            return "Pinch harmonics and dive-bomb tremolo accents punctuate the riff breaks"
        if archetype == "indie-pop" and beat_reg > 0.85:
            return "A shaker and tambourine keep a buoyant sixteenth-note pulse through the verses"
        if archetype == "rnb" and dance > 0.60:
            return "Finger snaps on the backbeat add a minimal, cool-jazz touch"
        if archetype in ("folk-rock",) and perc_ratio > 0.15:
            return "Handclaps accentuate the backbeat in the choruses"
        if dance > 0.75 and bpm > 105:
            return "Handclaps accent the backbeat through the chorus sections"
        return None

    # ────────────────────────────────────────────────────────────────────────
    # STEP 9: ARRANGEMENT (cinematic description of WHAT YOU HEAR)
    # ────────────────────────────────────────────────────────────────────────
    def _arrangement_desc():
        labels = [s.get("label", "") for s in (structure if isinstance(structure, list) else [])]
        has_intro  = any("Intro"  in l for l in labels)
        has_bridge = any("Bridge" in l for l in labels)
        has_outro  = any("Outro"  in l for l in labels)
        has_verse  = any("Verse"  in l for l in labels)
        has_chorus = any("Chorus" in l for l in labels)
        has_break  = any("Break"  in l for l in labels)

        # Archetype-specific arrangement language
        bridge_desc = {
            "gospel-soul": "a stripped-back breakdown with spoken word over a minimal organ and handclap bed",
            "folk-rock":   "a breakdown with spoken word over a minimal acoustic guitar and organ bed",
            "alt-rock":    "a tension-building instrumental breakdown before the final cathartic release",
            "dark-alt":    "an oppressive, near-silent breakdown where only low bass and a single guitar note remain",
            "indie-pop":   "a dreamy, lo-fi bridge of layered vocals and strummed guitar",
            "rnb":         "a melismatic vocal ad-lib breakdown over sparse piano",
            "edm":         "a tension-building filter-sweep breakdown before the crushing final drop",
            "ambient":     "a meditative, near-silent passage of pure reverb and harmonic overtones",
        }.get(archetype, "a stripped-back bridge section over minimal instrumentation")

        outro_desc = {
            "gospel-soul": "a euphoric, gospel choir-led outro that fades into the heavens",
            "folk-rock":   "a warm acoustic outro resolving into comfortable, communal silence",
            "alt-rock":    "a cascading wall of feedback and distortion that slowly dissolves",
            "dark-alt":    "the arrangement suddenly cuts to complete silence on a final drum hit",
            "indie-pop":   "a shimmering guitar-driven outro fading into reverb-washed calm",
            "edm":         "the outro strips back to a lone hi-hat before full silence",
            "rnb":         "a slow, melismatic vocal outro fading over a sparse piano figure",
        }.get(archetype, "a resolving outro that gradually strips back to silence")

        parts = []
        if has_intro:  parts.append("an atmospheric, minimal intro that slowly builds in density")
        if has_verse:  parts.append("an intimate verse spotlighting the lead vocal")
        if has_chorus: parts.append("a full-band chorus with explosive energy")
        if has_break:  parts.append("an instrumental break offering dynamic contrast")
        if has_bridge: parts.append(bridge_desc)
        if has_outro:  parts.append(outro_desc)

        if len(parts) >= 3:
            return "The arrangement moves through {}, and {}".format(", ".join(parts[:-1]), parts[-1])
        if has_bridge:
            return "The arrangement includes {}".format(bridge_desc)
        if len(parts) == 2:
            return "The arrangement moves from {} to {}".format(parts[0], parts[1])
        return None

    # ────────────────────────────────────────────────────────────────────────
    # STEP 10: ASSEMBLE STYLE PROMPT
    # ────────────────────────────────────────────────────────────────────────
    lead_inst = _lead_instrument()
    bass_d    = _bass_desc()
    drum_d    = _drum_desc()
    texture_d = _texture_desc()
    vocal_d   = _vocal_desc()
    accent_d  = _accent_desc()
    arr_d     = _arrangement_desc()
    technical = "{:.0f} BPM, Key of {}, {} time".format(bpm, key_full, ts)

    clauses = [genre_str]

    # Instruments block (combine into one "Features..." sentence)
    instr_parts = [x for x in [lead_inst, bass_d, drum_d] if x]
    if len(instr_parts) >= 3:
        clauses.append("Features {}, {}, and {}".format(*instr_parts[:3]))
    elif len(instr_parts) == 2:
        clauses.append("Features {} and {}".format(*instr_parts))
    elif instr_parts:
        clauses.append("Features {}".format(instr_parts[0]))

    if texture_d:  clauses.append(texture_d)
    clauses.append(vocal_d)
    if accent_d:   clauses.append(accent_d)
    if arr_d:      clauses.append(arr_d)
    clauses.append(technical)

    style_prompt = ", ".join(clauses)

    # ────────────────────────────────────────────────────────────────────────
    # STRUCTURE PROMPT (Lyrics box) — cinematic descriptions
    # ────────────────────────────────────────────────────────────────────────
    SECTION_NOTES = {
        "gospel-soul": {
            "Chorus": "full band, organ swells, gospel choir in full voice, anthemic",
            "Verse":  "stripped back, intimate, vocal-forward storytelling",
            "Bridge": "minimal — spoken word over organ drone and handclaps",
            "Intro":  "organ alone, then bass enters, building to verse",
            "Outro":  "choir-led, euphoric fade into silence",
        },
        "folk-rock": {
            "Chorus": "full band, acoustic guitar strumming hard, choir harmonies",
            "Verse":  "guitar and voice only, close and personal",
            "Bridge": "spoken word over single guitar and organ bed",
            "Intro":  "solo acoustic fingerpicking, gentle build",
            "Outro":  "gradual strip-back to single guitar, warmly resolved",
        },
        "alt-rock": {
            "Chorus": "walls of distorted guitar, explosive dynamics",
            "Verse":  "quiet, tense, guitar barely audible",
            "Bridge": "instrumental chaos, feedback, tension",
            "Intro":  "ambient guitar texture, slow build",
            "Outro":  "noise, feedback, slow decay to silence",
        },
        "dark-alt": {
            "Chorus": "cold, mechanical, full band but emotionless",
            "Verse":  "minimal — bass, drum, spoken vocal only",
            "Bridge": "near silence — single sustained note",
            "Intro":  "low industrial drone, oppressive atmosphere",
            "Outro":  "sudden cut to complete silence",
        },
        "indie-pop": {
            "Chorus": "lush, layered guitars and vocals, bright and joyful",
            "Verse":  "sparse, dreamy, close-mic vocal",
            "Bridge": "lo-fi breakdown, layered vocal harmonies",
            "Intro":  "gentle guitar loop, shimmering and inviting",
            "Outro":  "fading reverb, gentle and bittersweet",
        },
        "edm": {
            "Chorus": "full drop — distorted kick, bass sidechained, euphoric lead synth",
            "Verse":  "build — filtered bass, rising arpeggios",
            "Bridge": "breakdown — filter sweep, tension rising",
            "Intro":  "minimal loop, atmospheric pads, crowd anticipation",
            "Outro":  "gradual strip to silence — lone hi-hat last",
        },
    }

    def _seg_note(label, dur):
        notes = SECTION_NOTES.get(archetype, {})
        base  = label.split()[0]  # "Chorus 2" -> "Chorus"
        note  = notes.get(base, "")
        dur_s = int(dur)
        return "({} — {}s)".format(note, dur_s) if note else "({}s)".format(dur_s)

    struct_lines = []
    for seg in (structure if isinstance(structure, list) else []):
        label = seg.get("label", "Section")
        dur   = seg.get("duration", 0)
        struct_lines.append("[{}]".format(label))
        struct_lines.append(_seg_note(label, dur))
        struct_lines.append("")
    structure_prompt = "\n".join(struct_lines)

    # ────────────────────────────────────────────────────────────────────────
    # CHORD HINT
    # ────────────────────────────────────────────────────────────────────────
    if chords["main_prog_4"] != "N/A":   chord_hint = chords["main_prog_4"]
    elif chords["main_prog_3"] != "N/A": chord_hint = chords["main_prog_3"]
    elif chords["top_chords"]:
        chord_hint = " -> ".join(c[0] for c in chords["top_chords"][:4]) + " (frequency order)"
    else:
        chord_hint = "{}{}".format(root, "m" if scale=="Minor" else "") + " (diatonic)"

    # ────────────────────────────────────────────────────────────────────────
    # ASSEMBLE FULL TEXT OUTPUT
    # ────────────────────────────────────────────────────────────────────────
    SEP = "\u2501" * 60
    lines = []
    A = lines.append
    A("=" * 60)
    A("SUNO AI PROMPT")
    A("Tai hien: {}".format(meta.get("title") or "Bai nhac"))
    if meta.get("artist"): A("Nghe si goc: {}".format(meta["artist"]))
    A("=" * 60)
    A("")
    A(SEP)
    A("BUOC 1 — Dan vao o [Style of Music] tren Suno")
    A(SEP)
    A("")
    A(style_prompt)
    A("")
    A(SEP)
    A("BUOC 2 — Dan vao o [Lyrics] tren Suno")
    A(SEP)
    A("")
    if lyrics and lyrics.strip():
        A(lyrics.strip())
    else:
        A(structure_prompt)
    A("")
    A(SEP)
    A("THONG TIN PHAN TICH (tham khao khi tinh chinh)")
    A(SEP)
    A("")
    A("Tong nhac:     {} (confidence: {}%)".format(key_full, key_info["confidence_pct"]))
    A("Tempo:         {} BPM / {}  [Archetype: {}]".format(bpm, ts, archetype))
    A("Vong hop am:  {}".format(chord_hint))
    A("Loudness:      {} LUFS".format(dynamics.get("loudness_lufs", "N/A")))
    A("Danceability:  {} — {}".format(extra["danceability_score"], extra["danceability_label"]))
    A("Mood/Valence:  {} — {}".format(extra["valence_score"], extra["valence_label"]))
    A("Am sac:        {}".format(timbre["brightness"]))
    A("Audio Tags:    {}".format(", ".join(audio_tags)))
    A("")
    A(SEP)
    A("HUONG DAN TINH CHINH STYLE PROMPT")
    A(SEP)
    A("")
    A("Thay Genre: doi phan dau prompt")
    A("  Folk-rock -> 'indie folk-rock', 'Americana roots-rock', 'country soul'")
    A("  Pop       -> 'synth-pop', 'bedroom pop', 'baroque pop'")
    A("  Soul/R&B  -> 'neo-soul', 'classic soul', 'Motown-influenced R&B'")
    A("  Rock      -> 'shoegaze', 'post-punk', 'noise rock', 'grunge'")
    A("  Dark      -> 'cold wave', 'dark ambient', 'industrial'")
    A("")
    A("Thay Nhac cu: doi mo ta cu the")
    A("  Guitar: 'slide guitar riffs', 'wiry Telecaster picking', 'open-G tuning'")
    A("  Organ:  'Leslie-cabinet organ swells', 'Vox Continental organ stabs'")
    A("  Piano:  'honky-tonk upright piano', 'Rhodes electric piano'")
    A("  Strings:'sweeping orchestral strings', 'sul ponticello bowed cello'")
    A("")
    A("Thay Vocal:")
    A("  'Raspy, smoke-cured baritone', 'crystalline soprano', 'androgynous falsetto'")
    A("  'overlapping rounds of vocal counterpoint', 'unison choir chanting'")
    A("")
    A("Suno Tips:")
    A("  * Cang mo ta cu the cang tot — Suno hieu 'eighth-note patterns' hon 'guitar'")
    A("  * Them 'studio quality, professional mix' vao cuoi style")
    A("  * Generate 4-5 lan, chon version tot nhat")
    A("  * (whispered), (shouted), (falsetto) trong lyrics de dinh huong giong")
    A("  * [Instrumental] neu khong muon loi")
    A("")
    A("=" * 60)
    return "\n".join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Ví dụ: python3 music_deep_analyzer.py bai_hat.mp3")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"[LỖI] Không tìm thấy file: {file_path}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  🎵 MUSIC DEEP ANALYZER")
    print(f"  File: {os.path.basename(file_path)}")
    print("=" * 60)
    print()

    # Report dependency status
    print("[Dependencies]")
    dep_status = {
        "librosa"   : "✅",
        "tinytag"   : "✅" if DEPS.get("tinytag")   else "⚠️  (pip install tinytag)",
        "pyloudnorm": "✅" if DEPS.get("pyloudnorm") else "⚠️  (pip install pyloudnorm)",
        "madmom"    : "✅" if DEPS.get("madmom")    else "⚠️  (pip install madmom — chord sẽ dùng fallback)",
        "scipy"     : "✅" if DEPS.get("scipy")     else "⚠️  (pip install scipy)",
    }
    for name, status in dep_status.items():
        print(f"  {name:<12}: {status}")
    print()

    start_time = time.time()

    # ── 1. Metadata ──
    progress("Đọc metadata...")
    meta = get_metadata(file_path)

    # ── 1.5. Separate Stems (Demucs) ──
    vocals_path, no_vocals_path = separate_stems(file_path)
    if vocals_path and no_vocals_path:
        progress("Tải file No-Vocals cho phân tích Harmonic...")
        y_harmonic, sr_harmonic = librosa.load(no_vocals_path, sr=None, mono=True)
    else:
        y_harmonic, sr_harmonic = None, None

    # ── 2. Load Audio ──
    y, sr, duration, channels = load_audio(file_path)

    file_size_mb = round(os.path.getsize(file_path) / 1024 / 1024, 2)
    audio_props  = {
        "duration_sec": round(duration, 2),
        "duration_str": fmt_time(duration),
        "sample_rate" : sr,
        "total_samples": len(y),
        "channels"    : channels,
        "file_size_mb": file_size_mb
    }

    # ── 3. Rhythm ──
    # Nhịp điệu tính trên file gốc (có trống và bass rõ nhất)
    rhythm = analyze_rhythm(y, sr)

    # ── 4. Key ──
    # Tông nhạc tính trên file no_vocals (nếu có) để chuẩn hơn
    key_info = analyze_key(y_harmonic if y_harmonic is not None else y, sr_harmonic if sr_harmonic else sr)

    # ── 5. Chords ──
    # Hợp âm tính trên file no_vocals (nếu có) để chuẩn hơn
    chords = analyze_chords(y_harmonic if y_harmonic is not None else y, sr_harmonic if sr_harmonic else sr, rhythm["beat_times"], key_info)

    # ── 6. Structure ──
    structure = analyze_structure(y, sr, duration, rhythm["beat_times"])

    # ── 7. Dynamics ──
    dynamics = analyze_dynamics(y, sr, duration)

    # ── 8. Timbre ──
    timbre = analyze_timbre(y, sr)

    # ── 9. Extra ──
    extra = analyze_extra(y, sr, duration)

    # ── 10. Instruments / Tags ──
    audio_tags = analyze_instruments(file_path)

    # ── 11. Lyrics ──
    # Lấy vocals_path nếu có để Whisper bóc băng tốt hơn
    target_lyric_audio = vocals_path if vocals_path else file_path
    lyrics = analyze_lyrics(target_lyric_audio)
    
    # ── 12. Cleanup ──
    cleanup_stems()

    elapsed = round(time.time() - start_time, 1)
    progress(f"Hoàn thành phân tích sau {elapsed}s")
    print()

    # ── Build outputs ──
    report_md    = build_report(file_path, meta, audio_props, rhythm, key_info,
                                chords, structure, dynamics, timbre, extra, audio_tags, lyrics)
    suno_prompt  = build_suno_prompt(meta, rhythm, key_info, chords, structure,
                                     dynamics, timbre, extra, audio_tags, lyrics)

    # ── Save outputs ──
    base_name = os.path.splitext(file_path)[0]
    report_path = base_name + "_analysis.md"
    prompt_path = base_name + "_suno_prompt.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(suno_prompt)

    # ── Print summary ──
    print("=" * 60)
    print("  ✅ HOÀN THÀNH!")
    print("=" * 60)
    print()
    print(f"📊 BÁO CÁO PHÂN TÍCH:")
    print(f"   {report_path}")
    print()
    print(f"🎵 SUNO AI PROMPT:")
    print(f"   {prompt_path}")
    print()
    print("─" * 60)
    print("SUNO PROMPT (preview):")
    print("─" * 60)
    # In 30 dòng đầu của suno prompt
    preview_lines = suno_prompt.split("\n")[:35]
    print("\n".join(preview_lines))
    print("...")
    print()

if __name__ == "__main__":
    main()
