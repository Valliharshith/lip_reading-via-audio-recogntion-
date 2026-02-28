# audio/audio_buffer.py
import sounddevice as sd
import numpy as np
import threading
from collections import deque
from config import SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_CHUNK_DURATION

class AudioBuffer:
    """Continuous microphone capture with thread-safe ring buffer."""
    def __init__(self):
        self.buffer = deque(
            maxlen=int(SAMPLE_RATE * AUDIO_CHUNK_DURATION * 4)
        )
        self.running = False
        self.lock = threading.Lock()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        with self.lock:
            self.buffer.extend(indata[:, 0].tolist())

    def start(self):
        self.running = True
        self.stream = sd.InputStream(device=1,
            samplerate=SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            callback=self._callback,
            blocksize=1024,
            dtype='float32'
        )
        self.stream.start()

    def get_chunk(self, duration=None):
        """Get latest N seconds of audio as numpy array."""
        duration = duration or AUDIO_CHUNK_DURATION
        n_samples = int(SAMPLE_RATE * duration)
        with self.lock:
            data = list(self.buffer)
        if len(data) < n_samples:
            # Pad with zeros if not enough audio yet
            pad = [0.0] * (n_samples - len(data))
            data = pad + data
        return np.array(data[-n_samples:], dtype=np.float32)

    def is_speech(self, threshold=0.005):
        """Simple energy-based VAD."""
        chunk = self.get_chunk(duration=0.5)
        energy = np.sqrt(np.mean(chunk**2))
        return energy > threshold

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()