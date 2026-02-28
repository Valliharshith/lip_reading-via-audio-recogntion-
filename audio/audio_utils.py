import numpy as np
import librosa

def preprocess_audio(audio_bytes, original_sr=48000):
    # bytes → int16
    audio = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

    # stereo → mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # resample to 16000
    audio = librosa.resample(audio, orig_sr=original_sr, target_sr=16000)

    return audio