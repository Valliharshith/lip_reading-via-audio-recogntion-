# video/frame_buffer.py
import cv2
import threading
import time
from collections import deque
from config import WEBCAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT

class FrameBuffer:
    def __init__(self, maxlen=60):
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.buffer = deque(maxlen=maxlen)
        self.running = False
        self.thread = None
        self.fps = 0
        self._last_time = time.time()
        self._frame_count = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.buffer.append(frame)
                self._frame_count += 1
                now = time.time()
                elapsed = now - self._last_time
                if elapsed >= 1.0:
                    self.fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._last_time = now

    def get_latest(self):
        if self.buffer:
            return self.buffer[-1].copy()
        return None

    def get_recent_frames(self, n=15):
        frames = list(self.buffer)
        return frames[-n:] if len(frames) >= n else frames

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.cap.release()
