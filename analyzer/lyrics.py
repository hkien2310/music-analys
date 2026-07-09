"""
lyrics.py — Bóc băng lời bài hát bằng OpenAI Whisper.

Sử dụng Whisper model "base" để transcribe audio thành text.
Xử lý graceful khi Whisper chưa được cài đặt.
"""

from .utils import progress

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
