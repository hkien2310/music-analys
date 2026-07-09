"""
audio_loader.py — Tải và xử lý file âm thanh.

Bao gồm:
- load_audio: Load audio file, trả về mono/stereo, sample rate, duration
- separate_stems: Tách âm thanh bằng Demucs (vocals/no_vocals)
- cleanup_stems: Dọn dẹp file tạm sau khi tách stems
"""

import librosa

from .utils import progress

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
