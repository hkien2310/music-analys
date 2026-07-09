"""
suno_prompt_builder.py — Tạo ra file văn bản chứa prompt copy-paste trực tiếp cho Suno AI.

Extracted from music_deep_analyzer.py lines 1170-1695 with minimal changes:
  - Imports changed to relative package imports
  - All logic, archetype detection, genre strings, instrument/vocal/accent/arrangement
    descriptions, section notes, chord hint logic, and full text output assembly
    are EXACTLY preserved.
"""

from .config import PITCH_CLASSES
from .utils import fmt_time, progress


def build_suno_prompt(meta, rhythm, key_info, chords, structure, dynamics, timbre, extra, audio_tags, lyrics="", archetype_result=None):
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

    if archetype_result:
        archetype = archetype_result["archetype"]
    else:
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

    # Use archetype_result's genre_string if available, otherwise fallback to GENRE_STRINGS
    if archetype_result and archetype_result.get("genre_string"):
        genre_str = archetype_result["genre_string"]
    else:
        genre_str = GENRE_STRINGS.get(archetype, "Contemporary pop with alternative influences")

    # If scale is Minor, skew toward darker vocabulary
    if scale == "Minor" and "soul" in genre_str:
        genre_str = genre_str.replace("warmth", "melancholy")
    if scale == "Minor" and archetype == "indie-pop":
        genre_str = "Melancholic indie pop with dream-pop and shoegaze influences"

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: LEAD RHYTHM INSTRUMENT (specific technique + sonic character)
    # ────────────────────────────────────────────────────────────────────────
    def _lead_instrument():
        tags_lower = [t.lower() for t in audio_tags]
        detected_leads = []
        if "piano" in tags_lower: detected_leads.append("a melodic piano")
        if "acoustic guitar" in tags_lower: detected_leads.append("an acoustic guitar")
        if "electric guitar" in tags_lower: detected_leads.append("an electric guitar")
        if "synthesizer" in tags_lower: detected_leads.append("a lead synthesizer")
        if "saxophone" in tags_lower: detected_leads.append("a saxophone")
        if "whistling" in tags_lower or "whistle" in tags_lower: detected_leads.append("a prominent whistling melody")
        
        if detected_leads:
            return " and ".join(detected_leads) + " leading the arrangement"

        if archetype == "folk-rock":
            return "a strummed acoustic guitar"
        if archetype == "indie-pop":
            return "an electric guitar with clean arpeggios"
        if archetype in ("gospel-soul", "jazz", "rnb") and harm_ratio > 0.8:
            return "a warm piano"
        if archetype == "alt-rock":
            return "a distorted rhythm guitar"
        if archetype == "dark-alt":
            return "a dark, heavily chorused bassline"
        if archetype == "metal":
            return "a heavily distorted, down-tuned electric guitar playing aggressive riffs"
        if archetype == "contemporary-pop" and bpm > 110:
            return "a bright, plucky synthesizer lead"
        if archetype == "ambient":
            return "a lush, slowly evolving synthesizer pad"
        if archetype == "edm":
            return "a massive, detuned supersaw synthesizer"
        
        return "a steady rhythm section"

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
        tags_lower = [t.lower() for t in audio_tags]
        if "choir" in tags_lower:
            backing = "supported by a large choir providing harmonized backing"
        elif archetype in ("gospel-soul",) or (valence > 0.60 and dance > 0.65):
            backing = "supported by upbeat harmonized backing vocals"
        elif archetype in ("folk-rock", "blues"):
            backing = "backed by warm three-part harmonies"
        elif archetype in ("indie-pop", "contemporary-pop") and valence > 0.50:
            backing = "supported by layered vocal harmonies"
        elif archetype in ("alt-rock", "dark-alt"):
            backing = "with sparse, dissonant backing harmonies"
        elif archetype == "rnb":
            backing = "with breathy, interlocking backing vocals"
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
    
    # Inject actual Audio Tags explicitly for Suno
    real_tags = [t for t in audio_tags if t.lower() not in ("music", "speech", "thunk", "inside, small room")]
    if real_tags:
        clauses.append("Prominent elements: " + ", ".join(real_tags))

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
    conf_str = f" ({archetype_result['confidence']:.0%})" if archetype_result else ""
    A("Tempo:         {} BPM / {}  [Archetype: {}{}]".format(bpm, ts, archetype, conf_str))
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
