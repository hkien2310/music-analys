"""
archetype.py — Phát hiện phong cách âm nhạc (genre/archetype) bằng weighted scoring.

Kết hợp 3 lớp tín hiệu:
  Lớp 1 (trọng số cao): PANNs genre tags + probabilities  
  Lớp 2 (trung bình):   PANNs mood tags
  Lớp 3 (tie-breaker):  Audio features (BPM, spectral, harmonic ratio)
"""

from collections import defaultdict

# ─── PANNS GENRE TAG → ARCHETYPE SCORING ────────────────────────────────────
# Mỗi PANNs label đóng góp điểm cho 1+ archetype, nhân với probability

PANNS_GENRE_MAP = {
    # ── Pop ──
    "Pop music":                {"contemporary-pop": 3.0, "indie-pop": 1.0, "upbeat-pop": 1.0},
    # ── Rock ──
    "Rock music":               {"alt-rock": 3.0, "indie-pop": 0.5},
    "Rock and roll":            {"alt-rock": 2.5, "blues": 1.0},
    "Progressive rock":         {"alt-rock": 2.0, "metal": 1.5},
    "Psychedelic rock":         {"alt-rock": 2.5, "dark-alt": 1.5},
    "Punk rock":                {"alt-rock": 2.5, "metal": 1.5},
    # ── Indie ──
    "Independent music":        {"indie-pop": 3.5, "alt-rock": 1.0, "singer-songwriter": 1.0},
    # ── Folk / Country ──
    "Folk music":               {"folk-rock": 3.5, "singer-songwriter": 2.0, "indie-pop": 0.5},
    "Country":                  {"country": 3.5, "folk-rock": 2.0},
    # ── Soul / R&B / Gospel ──
    "Soul music":               {"gospel-soul": 3.5, "rnb": 2.0},
    "Gospel music":             {"gospel-soul": 4.0},
    "Rhythm and blues":         {"rnb": 4.0, "gospel-soul": 1.0},
    # ── Jazz / Blues ──
    "Jazz":                     {"jazz": 4.0, "rnb": 1.0},
    "Swing music":              {"jazz": 3.0},
    "Blues":                     {"blues": 4.0, "folk-rock": 1.0},
    # ── Hip-hop ──
    "Hip hop music":            {"hiphop": 4.0},
    "Rapping":                  {"hiphop": 3.5},
    # ── Electronic ──
    "Electronic dance music":   {"edm": 4.0},
    "Electronic music":         {"edm": 2.5, "ambient": 1.0},
    "Electronica":              {"edm": 2.0, "ambient": 1.5},
    "House music":              {"edm": 3.0, "upbeat-pop": 1.0},
    "Techno":                   {"edm": 3.5},
    "Trance music":             {"edm": 3.0, "ambient": 1.0},
    "Drum and bass":            {"edm": 3.0},
    "Dance music":              {"edm": 2.0, "upbeat-pop": 2.0},
    "Disco":                    {"upbeat-pop": 3.0, "edm": 1.5},
    # ── Ambient / Cinematic ──
    "Ambient music":            {"ambient": 4.0},
    "New-age music":            {"ambient": 3.0},
    "Soundtrack music":         {"ambient": 2.0, "contemporary-pop": 1.0},
    # ── Metal ──
    "Heavy metal":              {"metal": 4.0},
    # ── Other ──
    "Reggae":                   {"reggae": 4.0},
    "Salsa music":              {"latin": 3.5},
    "Classical music":          {"classical": 4.0, "ambient": 1.0},
}

# ─── PANNS MOOD TAG → SCORING + VALENCE MODIFIER ────────────────────────────

PANNS_MOOD_MAP = {
    "Happy music":    {"scores": {"upbeat-pop": 1.5, "contemporary-pop": 0.5},          "valence_boost": +0.25},
    "Sad music":      {"scores": {"dark-alt": 1.5, "ambient": 1.0, "singer-songwriter": 1.0}, "valence_boost": -0.25},
    "Tender music":   {"scores": {"folk-rock": 1.0, "singer-songwriter": 1.5, "ambient": 0.5}, "valence_boost": -0.10},
    "Exciting music": {"scores": {"upbeat-pop": 1.5, "edm": 1.0, "metal": 0.5},        "valence_boost": +0.15},
    "Angry music":    {"scores": {"metal": 2.0, "alt-rock": 1.5, "dark-alt": 1.0},     "valence_boost": -0.20},
    "Scary music":    {"scores": {"dark-alt": 2.0, "ambient": 1.0},                     "valence_boost": -0.20},
    "Funny music":    {"scores": {"upbeat-pop": 1.0},                                    "valence_boost": +0.10},
}

# ─── PANNS INSTRUMENT TAG → ARCHETYPE HINTS ─────────────────────────────────

PANNS_INSTRUMENT_MAP = {
    "Acoustic guitar":      {"folk-rock": 1.5, "singer-songwriter": 2.0, "country": 1.0},
    "Electric guitar":      {"alt-rock": 1.5, "indie-pop": 0.5, "blues": 0.5},
    "Steel guitar, slide guitar": {"blues": 2.0, "country": 2.0, "folk-rock": 1.0},
    "Piano":                {"contemporary-pop": 0.5, "jazz": 0.5, "singer-songwriter": 0.5},
    "Electric piano":       {"rnb": 1.0, "jazz": 1.0},
    "Synthesizer":          {"edm": 1.5, "contemporary-pop": 0.5, "indie-pop": 0.3},
    "Drum machine":         {"edm": 1.5, "hiphop": 1.0, "upbeat-pop": 0.5},
    "Hammond organ":        {"gospel-soul": 1.5, "blues": 1.0},
    "Organ":                {"gospel-soul": 1.0, "blues": 0.5},
    "Saxophone":            {"jazz": 2.0, "blues": 1.0, "rnb": 0.5},
    "Trumpet":              {"jazz": 1.5, "latin": 1.0},
    "Violin, fiddle":       {"classical": 1.5, "folk-rock": 1.0, "country": 1.0},
    "Harmonica":            {"blues": 2.0, "folk-rock": 1.5, "country": 1.0},
    "Flute":                {"classical": 1.0, "folk-rock": 0.5, "ambient": 0.5},
    "Double bass":          {"jazz": 1.5, "blues": 1.0},
    "Bass guitar":          {"alt-rock": 0.5, "rnb": 0.5},
    "Vibraphone":           {"jazz": 1.5, "ambient": 0.5},
}

# ─── PANNS VOCAL TAG → HINTS ────────────────────────────────────────────────

PANNS_VOCAL_MAP = {
    "Male singing":         {"alt-rock": 0.3, "folk-rock": 0.3, "hiphop": 0.2},
    "Female singing":       {"contemporary-pop": 0.3, "indie-pop": 0.3, "rnb": 0.2},
    "Child singing":        {"contemporary-pop": 0.2},
    "Choir":                {"gospel-soul": 1.5, "ambient": 0.5},
    "Synthetic singing":    {"edm": 1.0, "contemporary-pop": 0.5},
}

# ─── METADATA KEYWORD → ARCHETYPE (ưu tiên nếu có) ─────────────────────────

METADATA_KW_MAP = {
    "hip hop": "hiphop", "hip-hop": "hiphop", "rap": "hiphop",
    "r&b": "rnb", "soul": "gospel-soul", "gospel": "gospel-soul",
    "jazz": "jazz", "blues": "blues",
    "metal": "metal", "heavy metal": "metal",
    "ambient": "ambient", "classical": "classical",
    "electronic": "edm", "edm": "edm", "techno": "edm", "house": "edm",
    "folk": "folk-rock", "country": "country", "bluegrass": "folk-rock",
    "rock": "alt-rock", "punk": "alt-rock", "grunge": "alt-rock",
    "indie": "indie-pop", "lo-fi": "indie-pop", "lofi": "indie-pop",
    "reggae": "reggae", "ska": "reggae",
    "disco": "upbeat-pop", "funk": "upbeat-pop",
    "latin": "latin", "salsa": "latin", "bossa": "jazz",
    "singer-songwriter": "singer-songwriter", "singer songwriter": "singer-songwriter",
    "dream pop": "dream-pop", "dream-pop": "dream-pop",
    "shoegaze": "dream-pop",
}

# ─── ARCHETYPE DISPLAY NAMES ────────────────────────────────────────────────

ARCHETYPE_NAMES = {
    "contemporary-pop":  ("Contemporary pop",                "Pop đương đại"),
    "indie-pop":         ("Indie pop",                       "Indie pop"),
    "dream-pop":         ("Dream pop",                       "Dream pop"),
    "upbeat-pop":        ("Upbeat pop / Dance-pop",          "Pop sôi nổi"),
    "singer-songwriter": ("Singer-songwriter",               "Singer-songwriter"),
    "folk-rock":         ("Folk-rock / Americana",           "Folk-rock / Americana"),
    "country":           ("Country",                          "Country"),
    "alt-rock":          ("Alternative rock",                "Alternative rock"),
    "dark-alt":          ("Dark alternative / Post-punk",    "Dark alternative"),
    "gospel-soul":       ("Gospel-soul",                     "Gospel-soul"),
    "rnb":               ("Neo-soul / R&B",                  "Neo-soul / R&B"),
    "blues":             ("Blues",                            "Blues"),
    "jazz":              ("Jazz",                             "Jazz"),
    "hiphop":            ("Hip-hop",                         "Hip-hop"),
    "edm":               ("Electronic / EDM",                "Electronic / EDM"),
    "ambient":           ("Ambient / Cinematic",             "Ambient / Cinematic"),
    "metal":             ("Heavy metal",                     "Metal"),
    "reggae":            ("Reggae",                          "Reggae"),
    "latin":             ("Latin",                            "Latin"),
    "classical":         ("Classical",                        "Cổ điển"),
}

# ─── GENRE STRINGS FOR SUNO PROMPT ──────────────────────────────────────────

GENRE_STRINGS = {
    "gospel-soul":       "Gospel-soul with indie-folk warmth",
    "folk-rock":         "Indie folk-rock with Americana and roots-rock influences",
    "alt-rock":          "Brooding alternative rock with post-punk undertones",
    "dark-alt":          "Dark alternative rock with gothic and shoegaze undertones",
    "indie-pop":         "Sunlit indie pop with bedroom-pop and dream-pop influences",
    "dream-pop":         "Ethereal dream pop with shoegaze textures and lush reverb",
    "contemporary-pop":  "Contemporary pop with alternative and soul influences",
    "upbeat-pop":        "Contemporary pop with funk and gospel energy",
    "singer-songwriter": "Intimate singer-songwriter with acoustic folk sensibility",
    "country":           "Modern country with Americana roots and storytelling",
    "rnb":               "Smooth neo-soul with classic R&B and jazz influences",
    "edm":               "Electronic dance music with progressive house influences",
    "hiphop":            "Hip-hop with boom-bap and soul sample influences",
    "ambient":           "Cinematic ambient with post-rock and orchestral influences",
    "jazz":              "Jazz-influenced indie with sophisticated harmonic textures",
    "blues":             "Blues-rock with Southern soul and gospel influences",
    "metal":             "Heavy metal with progressive and alternative influences",
    "reggae":            "Reggae with dub influences and offbeat riddim",
    "latin":             "Latin pop with rhythmic percussion and melodic warmth",
    "classical":         "Orchestral classical with cinematic and romantic influences",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_archetype(audio_tags, tag_probs, rhythm, key_info, timbre, extra, meta):
    """
    Phát hiện phong cách âm nhạc bằng weighted scoring system.
    
    Args:
        audio_tags:  list[str]         — top display tags (for reference)
        tag_probs:   dict[str, float]  — ALL PANNs tags with probabilities
        rhythm:      dict              — from analyze_rhythm()
        key_info:    dict              — from analyze_key()
        timbre:      dict              — from analyze_timbre()
        extra:       dict              — from analyze_extra()
        meta:        dict              — from get_metadata()
    
    Returns:
        dict with keys:
            archetype       — str, primary archetype ID
            confidence      — float 0-1, how confident we are
            genre_label     — str, English display name
            genre_label_vn  — str, Vietnamese display name
            genre_string    — str, evocative genre string for Suno prompt
            mood            — str, detected mood from PANNs
            valence_adj     — float, valence adjustment from mood tags
            scores          — dict, all archetype scores (for debugging)
            top3            — list of (archetype, score) tuples
    """
    scores = defaultdict(float)
    valence_adj = 0.0
    detected_mood = ""
    
    # ── Extract features ──
    bpm         = rhythm.get("tempo_bpm", 120)
    scale       = key_info.get("scale", "Major")
    sc_hz       = timbre.get("spectral_centroid_hz", 2000)
    harm_ratio  = timbre.get("harmonic_ratio", 0.7)
    perc_ratio  = timbre.get("percussive_ratio", 0.3)
    valence     = extra.get("valence_score", 0.5)
    dance       = extra.get("danceability_score", 0.5)
    genre_raw   = (meta.get("genre") or "").lower().strip()
    
    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 0: METADATA GENRE (nếu có — override mạnh)
    # ══════════════════════════════════════════════════════════════════════════
    metadata_match = None
    for kw, arch in METADATA_KW_MAP.items():
        if kw in genre_raw:
            metadata_match = arch
            scores[arch] += 5.0  # Rất cao — metadata do người tag
            break
    
    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1: PANNS GENRE TAGS (trọng số cao nhất)
    # ══════════════════════════════════════════════════════════════════════════
    if tag_probs:
        for tag, mapping in PANNS_GENRE_MAP.items():
            prob = tag_probs.get(tag, 0.0)
            if prob >= 0.01:  # Threshold thấp — PANNs prob thường nhỏ
                for arch, weight in mapping.items():
                    # Nhân weight với probability → score
                    scores[arch] += weight * min(prob * 10, 3.0)  # Cap prob effect
    
    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 2: PANNS MOOD TAGS
    # ══════════════════════════════════════════════════════════════════════════
    if tag_probs:
        best_mood_prob = 0.0
        for tag, config in PANNS_MOOD_MAP.items():
            prob = tag_probs.get(tag, 0.0)
            if prob >= 0.01:
                # Add genre scores
                for arch, weight in config.get("scores", {}).items():
                    scores[arch] += weight * min(prob * 8, 2.0)
                # Track strongest mood for valence adjustment
                if prob > best_mood_prob:
                    best_mood_prob = prob
                    detected_mood = tag.replace(" music", "").lower()
                    valence_adj = config.get("valence_boost", 0.0) * min(prob * 5, 1.0)
    
    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 2.5: PANNS INSTRUMENT + VOCAL TAGS
    # ══════════════════════════════════════════════════════════════════════════
    if tag_probs:
        for tag, mapping in PANNS_INSTRUMENT_MAP.items():
            prob = tag_probs.get(tag, 0.0)
            if prob >= 0.01:
                for arch, weight in mapping.items():
                    scores[arch] += weight * min(prob * 6, 1.5)
        
        for tag, mapping in PANNS_VOCAL_MAP.items():
            prob = tag_probs.get(tag, 0.0)
            if prob >= 0.01:
                for arch, weight in mapping.items():
                    scores[arch] += weight * min(prob * 4, 1.0)
    
    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 3: AUDIO FEATURES (tie-breaker, trọng số thấp)
    # ══════════════════════════════════════════════════════════════════════════
    
    # BPM-based hints
    if bpm > 135 and perc_ratio > 0.35:
        scores["edm"] += 1.5
    elif bpm > 120 and perc_ratio > 0.30:
        scores["upbeat-pop"] += 1.0
    elif bpm < 85 and harm_ratio > 0.80:
        scores["ambient"] += 0.8
        scores["folk-rock"] += 0.6
    
    # Harmonic/Percussive balance
    if harm_ratio > 0.80 and bpm < 100:
        scores["folk-rock"] += 0.8
        scores["singer-songwriter"] += 0.6
    if perc_ratio > 0.40:
        scores["edm"] += 0.8
        scores["hiphop"] += 0.5
    
    # Scale hints
    if scale == "Minor":
        scores["dark-alt"] += 0.4
        scores["alt-rock"] += 0.3
    if scale == "Major":
        scores["contemporary-pop"] += 0.3
        scores["upbeat-pop"] += 0.2
    
    # Spectral characteristics
    if sc_hz < 1200 and harm_ratio > 0.60:
        scores["rnb"] += 0.8
    if sc_hz > 3500 and bpm > 120:
        scores["upbeat-pop"] += 0.5
    
    # Valence hints (minor role — PANNs mood is better)
    if valence < 0.35:
        scores["dark-alt"] += 0.5
        scores["ambient"] += 0.3
    if valence > 0.70 and dance > 0.70:
        scores["upbeat-pop"] += 0.5
        scores["gospel-soul"] += 0.3
    
    # ══════════════════════════════════════════════════════════════════════════
    # DETERMINE WINNER
    # ══════════════════════════════════════════════════════════════════════════
    
    if not scores:
        scores["contemporary-pop"] = 0.1  # Low score = low confidence for default
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner = sorted_scores[0]
    archetype = winner[0]
    max_score = winner[1]
    
    # Confidence: ratio of top score to sum of all scores
    total_score = sum(s for _, s in sorted_scores)
    if total_score > 0:
        ratio = max_score / total_score
        # Boost: if clear winner (big gap to #2), increase confidence
        runner_up = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        margin = (max_score - runner_up) / max_score if max_score > 0 else 0
        confidence = round(min(1.0, ratio * 0.7 + margin * 0.3), 2)
    else:
        confidence = 0.0
    
    # Get display names
    names = ARCHETYPE_NAMES.get(archetype, (archetype, archetype))
    genre_label = names[0]
    genre_label_vn = names[1]
    
    # Get genre string for Suno
    genre_string = GENRE_STRINGS.get(archetype, "Contemporary pop with alternative influences")
    
    # Minor key modifier on genre string
    if scale == "Minor" and "warmth" in genre_string:
        genre_string = genre_string.replace("warmth", "melancholy")
    if scale == "Minor" and archetype == "indie-pop":
        genre_string = "Melancholic indie pop with dream-pop and shoegaze influences"
    if scale == "Minor" and archetype == "contemporary-pop":
        genre_string = "Melancholic contemporary pop with introspective undertones"
    
    top3 = [(a, round(s, 2)) for a, s in sorted_scores[:3]]
    
    return {
        "archetype":      archetype,
        "confidence":     confidence,
        "genre_label":    genre_label,
        "genre_label_vn": genre_label_vn,
        "genre_string":   genre_string,
        "mood":           detected_mood,
        "valence_adj":    round(valence_adj, 3),
        "scores":         dict(scores),
        "top3":           top3,
    }
