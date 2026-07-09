"""
Music Deep Analyzer — Flask Web UI Backend
Chạy: source music_env/bin/activate && python app.py
Truy cập: http://localhost:5050
"""
import os, sys, tempfile, traceback, time, json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder="ui")
ALLOWED = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".opus"}


# ── Helper ────────────────────────────────────────────────────────────────────
def fmt_duration(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"

def clean(obj):
    """Serialize numpy types to plain Python."""
    import numpy as np
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):           return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [clean(v) for v in obj]
    return obj


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("ui", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "Không có file được gửi"}), 400
    f = request.files["file"]
    if not f.filename or Path(f.filename).suffix.lower() not in ALLOWED:
        return jsonify({"error": f"Định dạng không hỗ trợ: {', '.join(ALLOWED)}"}), 400

    suffix = Path(f.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        result = run_analysis(tmp_path, f.filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        try: os.unlink(tmp_path)
        except: pass


@app.route("/api/ai-enhance", methods=["POST"])
def ai_enhance():
    """
    Nhận analysis data + Gemini API key → trả về Styles + Lyrics đúng format Suno.
    Body JSON: { "api_key": "...", "analysis": {...}, "song_context": "..." }
    """
    data = request.get_json(silent=True) or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "Cần Gemini API key. Lấy miễn phí tại: https://aistudio.google.com/app/apikey"}), 400

    analysis = data.get("analysis", {})
    song_context = data.get("song_context", "")  # optional: user note about the song

    try:
        result = generate_with_gemini(api_key, analysis, song_context)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Gemini AI Enhancement ─────────────────────────────────────────────────────
def generate_with_gemini(api_key: str, analysis: dict, song_context: str = "") -> dict:
    """Call Gemini Flash to generate professional Suno STYLES + LYRICS."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # ── Build context from analysis ──
    r   = analysis.get("rhythm", {})
    k   = analysis.get("key", {})
    c   = analysis.get("chords", {})
    s   = analysis.get("structure", {})
    dyn = analysis.get("dynamics", {})
    tim = analysis.get("timbre", {})
    ex  = analysis.get("extra", {})
    meta = analysis.get("metadata", {})

    bpm      = r.get("bpm", 120)
    ts       = r.get("ts_label", "4/4")
    key_full = k.get("full", "C Major")
    scale    = k.get("scale", "Major")
    chord_prog = c.get("main_prog_4") or c.get("main_prog_3") or "N/A"
    top_chords = [x[0] if isinstance(x, list) else str(x) for x in c.get("top_chords", [])[:6]]
    segments   = s.get("segments", [])
    struct_str = " → ".join(seg.get("label","?") + f"({int(seg.get('duration',0))}s)" for seg in segments)

    valence     = ex.get("valence", 0.5)
    valence_lbl = ex.get("valence_label", "")
    dance       = ex.get("danceability", 0.5)
    dance_lbl   = ex.get("dance_label", "")
    lufs        = dyn.get("lufs", -14)
    harm_ratio  = tim.get("harmonic_ratio", 0.7)
    perc_ratio  = tim.get("percussive_ratio", 0.3)
    brightness  = tim.get("brightness_label", "")
    dom_notes   = tim.get("dominant_notes", [])
    song_name   = meta.get("title") or analysis.get("name", "Unknown")
    artist      = meta.get("artist", "")

    context_note = f"\nUser note about the song: {song_context}" if song_context else ""

    # ── System prompt ──
    system_prompt = """You are a world-class music producer and lyricist specializing in Suno AI prompts.
Your task: given detailed music analysis data, write a PROFESSIONAL Suno AI prompt in EXACTLY the format shown.

CRITICAL FORMAT RULES:
1. STYLES section = ONE flowing paragraph (no line breaks). 
   Must include: genre + influences, specific instruments with playing technique and sonic character,
   vocal description (include gender, quality, style), arrangement arc.
   Be evocative, specific, cinematic. Write like a real producer describing the track.
   
2. LYRICS section = Full song structure with Suno bracket headers.
   Header format: [Section Name | Mood/Energy | Technique cues]
   Example: [Verse 1 | Soft | Intimate | Talk-sung]
   IMPORTANT: You will be provided with the raw transcribed lyrics of the song. 
   You must distribute these original lyrics into the correct structural sections.
   DO NOT write placeholders. DO NOT invent fake lyrics. Just format the original lyrics neatly under the headers.
   Standard structure unless analysis shows otherwise: Intro → Verse 1 → Pre-Chorus → Chorus → Verse 2 → Pre-Chorus → Chorus → Bridge → Final Chorus → Outro

OUTPUT FORMAT (use exactly these markers):
===STYLES===
[your styles paragraph here]

===LYRICS===
[your full lyrics here]
"""

    user_msg = f"""MUSIC ANALYSIS DATA:
Song: {song_name}{" by " + artist if artist else ""}
Key: {key_full} (confidence: {k.get("confidence", 0)}%)
BPM: {bpm} | Time signature: {ts}
Chord progression: {chord_prog}
Top chords: {", ".join(top_chords)}
Song structure: {struct_str}
Harmonic ratio: {harm_ratio:.2f} (1.0 = pure melody, 0.0 = pure percussion)
Percussive ratio: {perc_ratio:.2f}
Timbre: {brightness}
Dominant musical notes: {", ".join(dom_notes)}
Valence (mood): {valence:.2f} — {valence_lbl}
Danceability: {dance:.2f} — {dance_lbl}
Loudness: {lufs:.1f} LUFS
Scale type: {scale} ({"bright, hopeful, energetic" if scale == "Major" else "emotional, introspective, tense"})
{context_note}

RAW TRANSCRIBED LYRICS:
{analysis.get('lyrics_raw', '(No lyrics found)')}

EXAMPLE OF PERFECT OUTPUT FORMAT:
===STYLES===
Acoustic singer-songwriter, fingerstyle acoustic guitar, upright bass, and female alto vocals, The acoustic guitar plays a gentle 3/4 waltz arpeggio pattern at 75 BPM in the key of D major, The upright bass enters in the first chorus with a deep, warm woody tone, The female alto lead vocal is intimate, breathy, and close-mic, singing with a quiet, maternal warmth, supported by two-part folk harmonies in the chorus, The arrangement is minimal and organic, focusing entirely on the intimacy of the story, stripping down to a single guitar in the bridge, A solo cello enters only at the final chorus, providing deep, emotional warmth as the final peak

===LYRICS===
[Intro]
(Gentle fingerpicked acoustic guitar, warm and intimate)

[Verse 1 | Soft | Intimate | Talk-sung]
Sample verse lyrics line one
Sample verse lyrics line two
Sample verse lyrics line three

[Chorus | Emotional peak | Full instrumentation]
Sample chorus lyrics
Sample chorus lyrics

Now write for the analyzed song above. Match the detected BPM ({bpm}), key ({key_full}), and mood ({valence_lbl}). Be creative with the lyrics — write an original story or theme that fits the sonic character."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.85,
            max_output_tokens=2500,
        ),
        contents=user_msg,
    )

    raw = response.text.strip()

    # ── Parse output ──
    styles = ""
    lyrics = ""

    if "===STYLES===" in raw and "===LYRICS===" in raw:
        parts = raw.split("===LYRICS===")
        styles_part = parts[0].split("===STYLES===")[-1].strip()
        lyrics_part = parts[1].strip() if len(parts) > 1 else ""
        styles = styles_part
        lyrics = lyrics_part
    else:
        # Fallback: first paragraph = styles, rest = lyrics
        lines = raw.split("\n")
        first_break = next((i for i, l in enumerate(lines) if l.strip() == "" and i > 0), 5)
        styles = " ".join(lines[:first_break]).strip()
        lyrics = "\n".join(lines[first_break:]).strip()

    # ── Save outputs to disk ──
    try:
        import os, re
        from datetime import datetime
        
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', song_name).strip('_')
        if not safe_name or safe_name == "Unknown_Song":
            safe_name = "Song_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", safe_name)
        os.makedirs(out_dir, exist_ok=True)
        
        with open(os.path.join(out_dir, "analysis.md"), "w", encoding="utf-8") as f:
            f.write(user_msg)
            
        with open(os.path.join(out_dir, "suno_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
            
        print(f"Saved outputs to {out_dir}")
    except Exception as e:
        print(f"Failed to save output files: {e}")

    return {
        "styles": styles,
        "lyrics": lyrics,
        "raw":    raw,
        "model":  "gemini-2.0-flash",
    }


# ── Analysis pipeline ────────────────────────────────────────────────────────
def run_analysis(audio_path: str, original_name: str = "") -> dict:
    from music_deep_analyzer import (
        get_metadata, load_audio, analyze_rhythm, analyze_key,
        analyze_chords, analyze_structure, analyze_dynamics,
        analyze_timbre, analyze_extra, analyze_lyrics, build_report, build_suno_prompt
    )

    name = Path(original_name).stem if original_name else Path(audio_path).stem
    t0 = time.time()

    meta = get_metadata(audio_path)
    y, sr, duration, channels = load_audio(audio_path)
    rhythm_raw = analyze_rhythm(y, sr)
    beat_times = rhythm_raw.get("beat_times", [])
    key_raw    = analyze_key(y, sr)
    chords_raw = analyze_chords(y, sr, beat_times, key_raw)
    struct_raw = analyze_structure(y, sr, duration, beat_times)
    dyn_raw    = analyze_dynamics(y, sr, duration)
    timbre_raw = analyze_timbre(y, sr)
    extra_raw  = analyze_extra(y, sr, duration)
    lyrics_raw = analyze_lyrics(audio_path)

    elapsed = round(time.time() - t0, 1)

    audio_props = {
        "duration":      duration,
        "duration_str":  fmt_duration(duration),
        "duration_sec":  round(duration, 1),
        "sample_rate":   int(sr),
        "sr":            int(sr),
        "channels":      int(channels),
        "total_samples": int(len(y)),
        "file_size_mb":  round(os.path.getsize(audio_path) / 1e6, 2),
    }
    report_md = build_report(audio_path, meta, audio_props,
                             rhythm_raw, key_raw, chords_raw,
                             struct_raw, dyn_raw, timbre_raw, extra_raw)
    suno_txt  = build_suno_prompt(meta, rhythm_raw, key_raw, chords_raw,
                                  struct_raw, dyn_raw, timbre_raw, extra_raw)

    bpm  = rhythm_raw.get("tempo_bpm", 0)
    feel = _bpm_feel(bpm)

    top_candidates  = key_raw.get("top_candidates", [])
    top5            = [{"key": f"{n} {s}", "score": c} for n, s, c in top_candidates[:5]]
    root            = key_raw.get("root", "?")
    scale           = key_raw.get("scale", "Major")
    parallel_label  = key_raw.get("relative_key", "")

    DIATONIC = {"Major": ["","m","m","","","m","dim"], "Minor": ["m","dim","","m","m","",""]}
    PITCH = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    SCALE_INTERVALS = {"Major": [0,2,4,5,7,9,11], "Minor": [0,2,3,5,7,8,10]}
    root_idx        = PITCH.index(root) if root in PITCH else 0
    intervals       = SCALE_INTERVALS.get(scale, SCALE_INTERVALS["Major"])
    suffixes        = DIATONIC.get(scale, DIATONIC["Major"])
    scale_notes     = [PITCH[(root_idx+i)%12] for i in intervals]
    diatonic_chords = [f"{PITCH[(root_idx+intervals[i])%12]}{suffixes[i]}" for i in range(len(intervals))]

    chord_sequence = chords_raw.get("sequence", [])
    total_dur = duration or 1
    chord_timeline = []
    for i, seg in enumerate(chord_sequence):
        nxt = chord_sequence[i+1]["start"] if i+1 < len(chord_sequence) else total_dur
        chord_timeline.append({
            "chord": seg["chord"],
            "start": seg["start"],
            "end":   round(nxt, 2),
            "start_str": seg.get("start_str", ""),
        })

    timeline = dyn_raw.get("energy_timeline", [])
    energy_frames = [{"time": f["time"], "time_str": fmt_duration(f["time"]), "rms": f["rms"]} for f in timeline]

    lufs      = dyn_raw.get("loudness_lufs")
    lufs_val  = lufs if lufs is not None else -20.0
    lufs_label = dyn_raw.get("loudness_note", "—")

    segments = []
    for s in (struct_raw if isinstance(struct_raw, list) else []):
        segments.append({
            "label":      s.get("label", "Section"),
            "start":      s.get("start", 0),
            "end":        s.get("end", 0),
            "duration":   s.get("duration", 0),
            "start_str":  s.get("start_str", ""),
            "end_str":    s.get("end_str", ""),
            "rms_ratio":  s.get("rms_ratio", 0),
            "energy_desc": s.get("energy_desc", ""),
        })

    return {
        "name": name, "elapsed": elapsed,
        "metadata": clean(meta),
        "audio": {
            "duration": round(duration, 1), "duration_str": fmt_duration(duration),
            "sr": int(sr), "channels": int(channels),
            "file_size": round(os.path.getsize(audio_path)/1e6, 2) if os.path.exists(audio_path) else 0,
        },
        "rhythm": {
            "bpm": bpm, "time_signature": int(rhythm_raw.get("time_signature","4/4").split("/")[0]),
            "total_beats": rhythm_raw.get("beats_total", 0),
            "beat_regularity": rhythm_raw.get("tempo_stability", 0),
            "onset_density": rhythm_raw.get("onset_density_per_sec", 0),
            "feel": feel, "ts_label": rhythm_raw.get("time_signature","4/4"),
            "ts_confidence": rhythm_raw.get("time_sig_confidence", ""),
        },
        "key": {
            "root": root, "scale": scale, "full": key_raw.get("full", f"{root} {scale}"),
            "confidence": key_raw.get("confidence_pct", 0), "parallel_label": parallel_label,
            "scale_notes": scale_notes, "diatonic_chords": diatonic_chords, "top5": top5,
        },
        "chords": {
            "main_prog_4": chords_raw.get("main_prog_4","N/A"),
            "main_prog_3": chords_raw.get("main_prog_3","N/A"),
            "top_chords":  clean(chords_raw.get("top_chords",[])),
            "unique_chords": chords_raw.get("unique_chords",0),
            "chord_timeline": chord_timeline,
        },
        "structure": {"segments": segments},
        "dynamics": {
            "rms_avg": dyn_raw.get("rms_avg",0), "rms_max": dyn_raw.get("rms_max",0),
            "dynamic_range": dyn_raw.get("dynamic_range_db",0),
            "lufs": lufs_val, "lufs_label": lufs_label, "energy_frames": energy_frames,
        },
        "timbre": {
            "spectral_centroid":  timbre_raw.get("spectral_centroid_hz",0),
            "spectral_bandwidth": timbre_raw.get("spectral_bandwidth_hz",0),
            "spectral_rolloff":   timbre_raw.get("spectral_rolloff_hz",0),
            "zcr":                timbre_raw.get("zero_crossing_rate",0),
            "harmonic_ratio":     timbre_raw.get("harmonic_ratio",0),
            "percussive_ratio":   timbre_raw.get("percussive_ratio",0),
            "brightness_label":   timbre_raw.get("brightness",""),
            "tonal_quality":      timbre_raw.get("tonal_quality",""),
            "hp_description":     timbre_raw.get("hp_description",""),
            "dominant_notes":     timbre_raw.get("dominant_notes",[]),
        },
        "extra": {
            "danceability": extra_raw.get("danceability_score",0),
            "dance_label":  extra_raw.get("danceability_label",""),
            "valence":      extra_raw.get("valence_score",0),
            "valence_label": extra_raw.get("valence_label",""),
        },
        "lyrics_raw": lyrics_raw,
        "report_md": report_md,
        "suno_txt":  suno_txt,
    }


def _bpm_feel(bpm: float) -> str:
    if bpm < 60:  return "Rất chậm (Grave) — ballad, ambient"
    if bpm < 76:  return "Chậm (Largo) — ballad, slow dance"
    if bpm < 96:  return "Vừa (Andante) — pop nhẹ nhàng, R&B"
    if bpm < 116: return "Sôi nổi (Allegretto) — pop, rock nhẹ"
    if bpm < 140: return "Nhanh (Allegro) — dance pop, rock"
    if bpm < 170: return "Rất nhanh (Presto) — metal, drum and bass"
    return "Cực nhanh — hardcore, speedcore"


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🎵 Music Deep Analyzer — Web UI")
    print("  Truy cập: http://localhost:5050")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
