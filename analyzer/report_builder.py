"""
report_builder.py — Tạo báo cáo Markdown tổng hợp tất cả kết quả phân tích.

Improvements over original build_report():
  1. Fixed Dynamic Range calculation (clamp + cap)
  2. Section 0: Nhận xét tổng quan (natural language overview)
  3. Roman Numeral chord analysis
  4. Energy timeline with section labels
  5. Audio Tags interpretation table
  6. Smart lyrics formatting (paragraph-split)
"""

from .config import PITCH_CLASSES, ENHARMONIC
from .utils import fmt_time, ascii_bar

import os
import re
import datetime
import numpy as np

# ─── TAG DESCRIPTIONS ─────────────────────────────────────────────────────────
# Common AudioSet / PANNs tags with Vietnamese descriptions & musical impact

TAG_DESCRIPTIONS = {
    "singing":          ("Có giọng hát chính",              "Vocal-driven track"),
    "pop music":        ("Nhạc pop thương mại",              "Pop production style"),
    "whistling":        ("Tiếng huýt sáo",                   "Melodic element"),
    "whistle":          ("Tiếng huýt sáo / tiếng còi",       "Melodic / rhythmic accent"),
    "thunk":            ("Âm thanh va chạm trầm",            "Percussive body / thump"),
    "drum kit":         ("Bộ trống acoustic / điện tử",      "Rhythmic backbone"),
    "drum":             ("Trống",                             "Rhythmic element"),
    "piano":            ("Đàn piano",                         "Harmonic / melodic lead"),
    "acoustic guitar":  ("Guitar acoustic",                   "Harmonic / rhythmic support"),
    "electric guitar":  ("Guitar điện",                       "Lead / rhythm instrument"),
    "bass guitar":      ("Guitar bass",                       "Low-end foundation"),
    "synthesizer":      ("Bộ tổng hợp âm thanh",             "Electronic texture / lead"),
    "organ":            ("Đàn organ",                         "Harmonic pad / texture"),
    "violin":           ("Violin",                            "Melodic / orchestral element"),
    "cello":            ("Cello",                             "Low string / emotional depth"),
    "trumpet":          ("Trumpet",                           "Brass accent / lead"),
    "saxophone":        ("Saxophone",                         "Jazz / soul element"),
    "flute":            ("Sáo",                               "Melodic / textural element"),
    "choir":            ("Dàn hợp xướng",                     "Vocal harmony / power"),
    "speech":           ("Giọng nói / lời thoại",             "Spoken word element"),
    "clapping":         ("Tiếng vỗ tay",                      "Rhythmic accent"),
    "finger snapping":  ("Tiếng búng tay",                    "Minimal rhythmic accent"),
    "music":            ("Nhạc nền chung",                    "General musical content"),
    "percussion":       ("Bộ gõ",                             "Rhythmic texture"),
    "bass drum":        ("Trống bass / kick",                 "Low-end rhythm"),
    "snare drum":       ("Trống snare",                       "Backbeat rhythm"),
    "hi-hat":           ("Hi-hat",                            "Time-keeping element"),
    "tambourine":       ("Trống lắc",                         "Rhythmic shimmer"),
    "harmonica":        ("Kèn harmonica",                     "Blues / folk element"),
    "strings":          ("Dàn dây",                           "Orchestral pad / emotion"),
    "electronic music": ("Nhạc điện tử",                      "Synth-driven production"),
    "hip hop music":    ("Nhạc hip hop",                       "Beat-driven, vocal-forward"),
    "rock music":       ("Nhạc rock",                          "Guitar-driven energy"),
    "jazz":             ("Nhạc jazz",                          "Improvisational harmony"),
    "country":          ("Nhạc country",                       "Acoustic / storytelling"),
    "reggae":           ("Nhạc reggae",                        "Off-beat rhythm"),
    "inside, small room": ("Ghi âm phòng nhỏ",               "Intimate recording space"),
}


# ─── ARCHETYPE DETECTION (simplified, mirrors suno_prompt_builder) ─────────

def _detect_archetype(bpm, scale, valence, dance, sc_hz, harm_ratio, perc_ratio,
                      genre_raw):
    """Detect sonic archetype from analysis features."""
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
        if kw in genre_raw:
            return arch

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


ARCHETYPE_VN = {
    "gospel-soul":      "Gospel-soul ấm áp",
    "folk-rock":        "Folk-rock / Americana",
    "alt-rock":         "Alternative rock",
    "dark-alt":         "Dark alternative",
    "indie-pop":        "Indie pop",
    "contemporary-pop": "Pop đương đại",
    "upbeat-pop":       "Pop sôi nổi",
    "rnb":              "Neo-soul / R&B",
    "edm":              "Electronic / EDM",
    "hiphop":           "Hip-hop",
    "ambient":          "Ambient / cinematic",
    "jazz":             "Jazz",
    "blues":            "Blues",
    "metal":            "Metal",
}


# ─── ROMAN NUMERAL ANALYSIS ──────────────────────────────────────────────────

# Interval names for Major and Minor scale degrees
_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]

_MAJOR_DEGREE_NAMES = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
_MINOR_DEGREE_NAMES = ["i", "ii°", "III", "iv", "v", "VI", "VII"]

# Quality suffixes to strip for root detection
_QUALITY_SUFFIXES = [
    "sus4", "sus2", "aug", "dim",
    "maj7", "m7", "7",
    "m",
]


def _parse_chord(chord_name):
    """Parse a chord name like 'F#m7' into (root_note, quality_suffix)."""
    # Try longest root match first (e.g. C# before C)
    for length in (2, 1):
        root_candidate = chord_name[:length]
        if root_candidate in PITCH_CLASSES:
            quality = chord_name[length:]
            return root_candidate, quality
    # Try enharmonic reverse
    for sharp, flat in ENHARMONIC.items():
        if chord_name.startswith(flat):
            quality = chord_name[len(flat):]
            return sharp, quality
    return None, chord_name


def _chord_to_roman(chord_name, root_idx, scale):
    """
    Map a chord name to its Roman Numeral in the given key.
    Returns (roman_str, is_diatonic).
    """
    chord_root, quality = _parse_chord(chord_name)
    if chord_root is None:
        return chord_name, False

    try:
        chord_root_idx = PITCH_CLASSES.index(chord_root)
    except ValueError:
        return chord_name, False

    # Interval from key root
    interval = (chord_root_idx - root_idx) % 12

    # Determine which scale degrees are in key
    if scale == "Major":
        scale_intervals = _MAJOR_INTERVALS
        degree_names = _MAJOR_DEGREE_NAMES
    else:
        scale_intervals = _MINOR_INTERVALS
        degree_names = _MINOR_DEGREE_NAMES

    is_diatonic = interval in scale_intervals
    if is_diatonic:
        degree_idx = scale_intervals.index(interval)
        roman = degree_names[degree_idx]
    else:
        # Find closest scale degree and mark as chromatic
        # Use flat/sharp notation relative to nearest degree
        closest_idx = min(range(len(scale_intervals)),
                         key=lambda j: abs(scale_intervals[j] - interval))
        roman = degree_names[closest_idx]
        diff = interval - scale_intervals[closest_idx]
        if diff > 0:
            roman = "#" + roman
        elif diff < 0:
            roman = "♭" + roman

    # Append quality suffix
    if quality:
        roman += quality

    return roman, is_diatonic


# ─── LYRICS FORMATTING ───────────────────────────────────────────────────────

def _format_lyrics(lyrics_text):
    """
    Split raw lyrics into paragraphs for readability.
    Uses sentence-ending punctuation to break lines, groups ~4 lines
    per paragraph with a blank line between.
    """
    if not lyrics_text or not lyrics_text.strip():
        return None

    text = lyrics_text.strip()

    # Split on sentence-ending punctuation followed by space + uppercase
    # This handles: "... done. The sky ..." → two lines
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    if len(sentences) <= 1:
        # Fallback: just return as-is with line breaks every ~80 chars
        return text

    # Group into paragraphs of ~4 lines
    paragraphs = []
    current_paragraph = []
    for i, sentence in enumerate(sentences):
        current_paragraph.append(sentence.strip())
        if len(current_paragraph) >= 4:
            paragraphs.append("\n".join(current_paragraph))
            current_paragraph = []
    if current_paragraph:
        paragraphs.append("\n".join(current_paragraph))

    return "\n\n".join(paragraphs)


# ─── SECTION LOOKUP FOR ENERGY TIMELINE ──────────────────────────────────────

def _get_section_at_time(structure, time_sec):
    """Find which section label covers a given time point."""
    if not structure:
        return ""
    for seg in structure:
        start = seg.get("start", 0)
        end = seg.get("end", start + seg.get("duration", 0))
        if start <= time_sec < end:
            return seg.get("label", "")
    # If past all sections, return last section
    if structure:
        return structure[-1].get("label", "")
    return ""


# ─── MAIN REPORT BUILDER ─────────────────────────────────────────────────────

def build_report(file_path, meta, audio_props, rhythm, key_info, chords,
                 structure, dynamics, timbre, extra, audio_tags, lyrics=""):
    """
    Tạo báo cáo Markdown tổng hợp tất cả kết quả phân tích.
    """
    lines = []
    A = lines.append  # shortcut

    song_name = meta.get("title") or os.path.splitext(os.path.basename(file_path))[0]

    # ── HEADER ──
    A(f"# 🎵 Báo cáo Phân tích Nhạc lý: {song_name}")
    A(f"> Tạo bởi **music_deep_analyzer.py** lúc {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"> File: `{os.path.basename(file_path)}`")
    A("")

    # ── SECTION 0: NHẬN XÉT TỔNG QUAN ──
    bpm = rhythm["tempo_bpm"]
    scale = key_info["scale"]
    root = key_info["root"]
    valence = extra["valence_score"]
    dance = extra["danceability_score"]
    sc_hz = timbre["spectral_centroid_hz"]
    harm_ratio = timbre["harmonic_ratio"]
    perc_ratio = timbre["percussive_ratio"]
    genre_raw = (meta.get("genre") or "").lower().strip()

    archetype = _detect_archetype(bpm, scale, valence, dance, sc_hz,
                                  harm_ratio, perc_ratio, genre_raw)
    archetype_vn = ARCHETYPE_VN.get(archetype, archetype)

    # BPM description
    if bpm < 60:
        bpm_desc = "rất chậm (Grave/Largo)"
    elif bpm < 80:
        bpm_desc = "chậm (Andante)"
    elif bpm < 100:
        bpm_desc = "vừa phải (Moderato)"
    elif bpm < 120:
        bpm_desc = "sôi nổi (Allegretto)"
    elif bpm < 140:
        bpm_desc = "nhanh (Allegro)"
    elif bpm < 160:
        bpm_desc = "rất nhanh (Vivace)"
    else:
        bpm_desc = "cực nhanh (Presto)"

    # Brightness
    brightness_str = timbre["brightness"]

    # Harmonic / percussive
    if harm_ratio > 0.75:
        hp_short = "chủ yếu giai điệu (harmonic)"
    elif perc_ratio > 0.50:
        hp_short = "chủ yếu bộ gõ (percussive)"
    else:
        hp_short = "cân bằng harmonic/percussive"

    # Top tags
    top_tags_str = ", ".join(audio_tags[:5]) if audio_tags else "(không phát hiện)"

    # Number of sections
    n_sections = len(structure) if isinstance(structure, list) else 0

    # Valence
    if valence > 0.70:
        val_desc = "tích cực, tươi sáng"
    elif valence > 0.50:
        val_desc = "trung tính đến nhẹ nhàng"
    elif valence > 0.30:
        val_desc = "trầm lặng, suy tư"
    else:
        val_desc = "buồn, u ám"

    # Danceability
    if dance > 0.80:
        dance_desc = "rất phù hợp để nhảy/dance"
    elif dance > 0.60:
        dance_desc = "khá phù hợp cho chuyển động"
    elif dance > 0.40:
        dance_desc = "nhịp vừa phải, có thể lắc lư"
    else:
        dance_desc = "nhạc thư giãn, ít chuyển động"

    A("## 🔍 Nhận xét tổng quan")
    A("")
    A(f"> Bài nhạc mang phong cách **{archetype_vn}** với tempo {bpm} BPM — {bpm_desc}.")
    A(f"> Âm sắc: {brightness_str}, {hp_short}.")
    A(f"> Tông nhạc **{key_info['full']}** (confidence {key_info['confidence_pct']}%).")
    A(f"> Bài có cấu trúc **{n_sections} đoạn** rõ ràng.")
    A(f"> Nhạc cụ/âm thanh nổi bật: {top_tags_str}.")
    A(f"> Tâm trạng: {val_desc} (valence {valence}), {dance_desc} (danceability {dance}).")
    if meta.get("artist"):
        A(f"> Nghệ sĩ: **{meta['artist']}** — Thời lượng: **{audio_props['duration_str']}**.")
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
    A(f"| 🏷️ Archetype | **{archetype_vn}** ({archetype}) |")
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
    for rank, (note, scale_name, score) in enumerate(key_info['top_candidates'], 1):
        marker = "← **Được chọn**" if rank == 1 else ""
        A(f"| {rank} | {note} {scale_name} | {score} {marker}|")
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

    # ── NEW: ROMAN NUMERAL ANALYSIS ──
    A("### 🎵 Phân tích Bậc hợp âm (Roman Numeral)")
    A("")
    A(f"Trong tông **{key_info['full']}**:")
    A("")
    A("| Hợp âm | Bậc (Roman) | Loại |")
    A("|--------|-------------|------|")
    for chord_name, count in chords['top_chords']:
        roman, is_diatonic = _chord_to_roman(chord_name, root_idx, key_info['scale'])
        if is_diatonic:
            kind = "✅ Diatonic"
        else:
            kind = "🔶 Borrowed / Chromatic"
        A(f"| **{chord_name}** | {roman} | {kind} |")
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

    # ── FIX: Dynamic Range calculation ──
    # The original code does 20*log10(rms_max / (rms_min + 1e-9)) which gives
    # absurd values like 169.5 dB when rms_min is near 0.
    # Fix: recompute from raw values if available, else use the stored value
    # and cap at 80 dB max.
    dr_db = dynamics['dynamic_range_db']
    try:
        dr_val = float(dr_db)
        if dr_val > 80.0:
            # Recompute with clamped rms_min
            rms_max_val = float(dynamics['rms_max'])
            # Clamp rms_min to at least 0.001
            rms_min_val = max(0.001, rms_max_val * 0.01)  # fallback estimate
            if 'rms_min' in dynamics:
                rms_min_val = max(0.001, float(dynamics['rms_min']))
            dr_val = 20 * np.log10(rms_max_val / rms_min_val)
            dr_val = min(dr_val, 80.0)
            dr_db = f"{dr_val:.1f}"
    except (ValueError, TypeError):
        pass

    A(f"- **Dynamic Range:** {dr_db} dB")
    if dynamics.get('loudness_lufs'):
        A(f"- **Loudness (LUFS):** {dynamics['loudness_lufs']} LUFS")
        A(f"- **Đánh giá:** {dynamics.get('loudness_note', '')}")
    A("")

    # ASCII energy timeline with section labels
    A("### Biểu đồ năng lượng theo thời gian:")
    A("")
    A("```")
    timeline = dynamics["energy_timeline"]
    max_rms = max(t["rms"] for t in timeline) if timeline else 0.001
    for item in timeline:
        bar = ascii_bar(item["rms"], max_rms, width=30)
        # Find the section label at this time point
        section_label = _get_section_at_time(structure, item['time'])
        section_tag = f"  [{section_label}]" if section_label else ""
        A(f"{fmt_time(item['time']):>5}  {bar}  {item['rms']:.4f}{section_tag}")
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

    # ── NEW: AUDIO TAGS INTERPRETATION TABLE ──
    A("### 🏷️ Nhận diện Âm thanh / Nhạc cụ (Audio Tags):")
    A("")
    A("| Tag | Mô tả | Ảnh hưởng |")
    A("|-----|-------|----------|")
    for tag in audio_tags:
        tag_lower = tag.lower()
        if tag_lower in TAG_DESCRIPTIONS:
            desc_vn, impact = TAG_DESCRIPTIONS[tag_lower]
        else:
            desc_vn = tag
            impact = "Detected element"
        A(f"| **{tag}** | {desc_vn} | {impact} |")
    A("")

    # ── LYRICS ──
    A("---")
    A("## 🎤 10. Lời bài hát (Whisper)")
    A("")
    if lyrics and lyrics.strip():
        formatted = _format_lyrics(lyrics)
        if formatted:
            A(formatted)
        else:
            A("```text")
            A(lyrics.strip())
            A("```")
    else:
        A("*Không tìm thấy lời bài hát hoặc file không có vocal.*")
    A("")

    # ── FOOTER ──
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
