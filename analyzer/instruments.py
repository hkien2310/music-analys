"""
instruments.py — Nhận diện nhạc cụ, giọng hát và không gian bằng PANNs (AudioSet).

Sử dụng PANNs (Audio Tagging) với model Cnn14 để phân loại
âm thanh và trả về các tag phù hợp nhất.
"""

from .utils import progress

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
