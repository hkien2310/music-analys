"""
config.py — Cấu hình, import dependencies, và hằng số cho analyzer package.

Bao gồm:
- Import & dependency detection (numpy, librosa, tinytag, pyloudnorm, madmom, scipy)
- Các hằng số nhạc lý: PITCH_CLASSES, ENHARMONIC, chord templates
- Profile Krumhansl-Schmuckler để detect key
- GENRE_MAP mapping metadata sang Suno tags
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
