#!/usr/bin/env python3
"""
music_deep_analyzer.py
======================
Phân tích bài nhạc cực kỳ chi tiết → báo cáo Markdown + Suno AI Prompt

Cách dùng:
    python3 music_deep_analyzer.py <file_âm_thanh.mp3>

Output:
    outputs/<tên_bài>/<tên_bài>_analysis.md    — Báo cáo nhạc lý đầy đủ
    outputs/<tên_bài>/<tên_bài>_suno_prompt.txt — Prompt tối ưu cho Suno AI
"""

import os
import sys
import time

# ─── Import analyzer package ──────────────────────────────────────────────────

from analyzer.config import DEPS
from analyzer.utils import progress, fmt_time
from analyzer.metadata import get_metadata
from analyzer.audio_loader import load_audio, separate_stems, cleanup_stems
from analyzer.rhythm import analyze_rhythm
from analyzer.key_detection import analyze_key
from analyzer.chord_analysis import analyze_chords
from analyzer.structure import analyze_structure
from analyzer.dynamics import analyze_dynamics
from analyzer.timbre import analyze_timbre
from analyzer.extra_features import analyze_extra
from analyzer.instruments import analyze_instruments
from analyzer.lyrics import analyze_lyrics
from analyzer.report_builder import build_report
from analyzer.suno_prompt_builder import build_suno_prompt

# ─── Thư viện cần thiết ──────────────────────────────────────────────────────

try:
    import librosa
except ImportError:
    print("[LỖI NGHIÊM TRỌNG] Thiếu librosa. Cài đặt: pip install librosa numpy")
    sys.exit(1)


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

    # ── Output Directory Setup ──
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join("outputs", base_name)
    os.makedirs(out_dir, exist_ok=True)

    stems_dir = os.path.join(out_dir, "stems")

    # ── 1.5. Separate Stems (Demucs) ──
    vocals_path, no_vocals_path = separate_stems(file_path, out_dir=stems_dir)
    if vocals_path and no_vocals_path:
        progress("Tải file No-Vocals cho phân tích Harmonic...")
        y_harmonic, sr_harmonic = librosa.load(no_vocals_path, sr=None, mono=True)
    else:
        y_harmonic, sr_harmonic = None, None

    # ── 2. Load Audio ──
    y, sr, duration, channels = load_audio(file_path)

    file_size_mb = round(os.path.getsize(file_path) / 1024 / 1024, 2)
    audio_props = {
        "duration_sec": round(duration, 2),
        "duration_str": fmt_time(duration),
        "sample_rate" : sr,
        "total_samples": len(y),
        "channels"    : channels,
        "file_size_mb": file_size_mb
    }

    # ── 3. Rhythm ──
    rhythm = analyze_rhythm(y, sr)

    # ── 4. Key ──
    key_info = analyze_key(
        y_harmonic if y_harmonic is not None else y,
        sr_harmonic if sr_harmonic else sr
    )

    # ── 5. Chords ──
    chords = analyze_chords(
        y_harmonic if y_harmonic is not None else y,
        sr_harmonic if sr_harmonic else sr,
        rhythm["beat_times"], key_info
    )

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
    target_lyric_audio = vocals_path if vocals_path else file_path
    lyrics = analyze_lyrics(target_lyric_audio)

    # ── 12. Cleanup ──
    cleanup_stems(stems_dir)

    elapsed = round(time.time() - start_time, 1)
    progress(f"Hoàn thành phân tích sau {elapsed}s")
    print()

    # ── Build outputs ──
    report_md = build_report(
        file_path, meta, audio_props, rhythm, key_info,
        chords, structure, dynamics, timbre, extra, audio_tags, lyrics
    )
    suno_prompt = build_suno_prompt(
        meta, rhythm, key_info, chords, structure,
        dynamics, timbre, extra, audio_tags, lyrics
    )

    # ── Save outputs ──
    report_path = os.path.join(out_dir, base_name + "_analysis.md")
    prompt_path = os.path.join(out_dir, base_name + "_suno_prompt.txt")

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
    preview_lines = suno_prompt.split("\n")[:35]
    print("\n".join(preview_lines))
    print("...")
    print()


if __name__ == "__main__":
    main()
