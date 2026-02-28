 # models/whisper_model.py
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config import WHISPER_MODEL, SAMPLE_RATE, DEVICE

# Realistic speech threshold based on your logs (noise ~0.01)
SPEECH_RMS_THRESHOLD = 0.002

class WhisperASR:
    """
    Wraps OpenAI Whisper-tiny for real-time CPU inference.
    Tiny model: ~39M params, ~2-4s latency on CPU per 3s chunk.
    """
    def __init__(self):
        print(f"[Whisper] Loading {WHISPER_MODEL}...")
        self.processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        self.model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
        self.model.eval()
        self.model.to(DEVICE)
        print("[Whisper] Ready ✅")

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray) -> str:
        """
        audio: float32 numpy array at 16kHz
        Returns: transcribed string
        """

        # ---- FIX 1: Ignore silence / background noise ----
        if audio is None or len(audio) == 0:
            return ""

        rms = float(np.sqrt(np.mean(audio**2)))

        # If user is not actually speaking → do NOT call Whisper
        if rms < SPEECH_RMS_THRESHOLD:
            return ""

        # Avoid decoding extremely short sounds (breath, click, lip smack)
        if len(audio) < SAMPLE_RATE * 0.5:
            return ""

        # ---- Normal Whisper inference ----
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(DEVICE)

        generated_ids = self.model.generate(
            input_features,
            max_new_tokens=50,
            num_beams=1,
            do_sample=False
        )

        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return transcription.strip()