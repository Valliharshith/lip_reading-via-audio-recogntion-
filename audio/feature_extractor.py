# audio/feature_extractor.py
import numpy as np
import librosa
from config import SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT

class AudioFeatureExtractor:
    def extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio, sr=SAMPLE_RATE,
            n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        return mfcc

    def extract_mel(self, audio):
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_mels=128, hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_db

    def extract_all(self, audio):
        return {
            "mfcc": self.extract_mfcc(audio),
            "mel": self.extract_mel(audio),
            "rms": float(np.sqrt(np.mean(audio**2))),
            "zcr": float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        }
