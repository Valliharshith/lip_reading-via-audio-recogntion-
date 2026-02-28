# inference/realtime_engine.py
import threading
import time
import numpy as np
from audio.audio_buffer import AudioBuffer
from audio.feature_extractor import AudioFeatureExtractor
from models.whisper_model import WhisperASR
from models.lip_motion_model import LipMotionDetector
from fusion.fusion_engine import FusionEngine
from config import AUDIO_CHUNK_DURATION

class RealtimeEngine:
    def __init__(self):
        self.audio_buf = AudioBuffer()
        self.audio_feat = AudioFeatureExtractor()
        self.asr = WhisperASR()
        self.lip_detector_model = LipMotionDetector()
        self.fusion = FusionEngine()
        self._lock = threading.Lock()
        self._transcription = ""
        self._audio_rms = 0.0
        self._running = False
        self._audio_thread = None

    def start(self):
        self.audio_buf.start()
        self._running = True
        self._audio_thread = threading.Thread(
            target=self._audio_inference_loop, daemon=True
        )
        self._audio_thread.start()
        print("[Engine] Started")

    def _audio_inference_loop(self):
        while self._running:
            t0 = time.time()
            audio_chunk = self.audio_buf.get_chunk()
            rms = float(np.sqrt(np.mean(audio_chunk**2)))

            if rms > 0.004:
                text = self.asr.transcribe(audio_chunk)
            else:
                text = ""

            print(f"[Audio] RMS={rms:.4f} | Result: '{text}'")

            with self._lock:
                self._audio_rms = rms
                if text:
                    self._transcription = text

            elapsed = time.time() - t0
            time.sleep(max(0, AUDIO_CHUNK_DURATION - elapsed))

    def process_frame(self, lip_crop):
        lip_state = self.lip_detector_model.get_lip_state(lip_crop)
        with self._lock:
            audio_text = self._transcription
            audio_rms = self._audio_rms
        result = self.fusion.fuse(audio_text, audio_rms, lip_state)
        result['lip_state'] = lip_state
        result['audio_rms'] = audio_rms
        return result

    def get_last_result(self):
        with self._lock:
            return {
                "text": self._transcription,
                "audio_rms": self._audio_rms
            }

    def stop(self):
        self._running = False
        self.audio_buf.stop()
        if self._audio_thread:
            self._audio_thread.join(timeout=5)
        print("[Engine] Stopped")
