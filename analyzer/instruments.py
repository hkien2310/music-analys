"""
instruments.py — Nhận diện nhạc cụ, giọng hát và không gian bằng PANNs (AudioSet).

Sử dụng PANNs (Audio Tagging) với model Cnn14 để phân loại
âm thanh và trả về các tag phù hợp nhất + toàn bộ probability dict.
"""

from .utils import progress

# ─── MODULE 1G: INSTRUMENTS & AUDIO TAGS (PANNs) ──────────────────────────────

def analyze_instruments(audio_path):
    """
    Sử dụng PANNs (AudioSet) để nhận diện nhạc cụ, giọng hát và không gian.
    
    Returns:
        tuple: (display_tags, tag_probs)
            - display_tags: list[str]       — top 6 tags cho hiển thị
            - tag_probs:    dict[str, float] — TẤT CẢ tags với probability (>= 0.005)
    """
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

        all_labels = np.array(labels)
        sorted_indexes = np.argsort(clipwise_output[0])[::-1]
        
        # ── Display tags: top tags cho UI/report ──
        display_tags = []
        ignore_tags = {
            "Music", "Musical instrument", "Sound", "Song", "Noise",
            "Inside, small room", "Inside, large room or hall", "Speech",
            "Music of Africa", "Music of Asia", "Music of Latin America"
        }

        for i in range(30):  # Scan nhiều hơn để có đủ tags tốt
            idx = sorted_indexes[i]
            label = all_labels[idx]
            prob = float(clipwise_output[0][idx])
            if label not in ignore_tags and prob >= 0.01:
                display_tags.append(label)
        
        display_tags = display_tags[:6]  # Chỉ hiển thị top 6

        # ── Full probability dict: cho archetype detection ──
        tag_probs = {}
        for i in range(len(all_labels)):
            prob = float(clipwise_output[0][i])
            if prob >= 0.005:  # Threshold rất thấp — giữ lại tín hiệu yếu
                tag_probs[all_labels[i]] = round(prob, 4)

        return display_tags, tag_probs
        
    except ImportError:
        return ["(panns-inference chưa được cài đặt)"], {}
    except Exception as e:
        progress(f"Lỗi khi chạy PANNs: {e}")
        return [], {}
