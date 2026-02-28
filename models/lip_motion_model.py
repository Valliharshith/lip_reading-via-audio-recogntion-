# models/lip_motion_model.py
import numpy as np

class LipMotionDetector:
    LABELS = ["silent", "speaking", "fast_motion"]

    def __init__(self):
        self._prev_frame = None
        print("[LipMotion] Ready")

    def optical_flow_motion(self, lip_crop):
        if lip_crop is None:
            return 0.0
        gray = np.mean(lip_crop, axis=2) if len(lip_crop.shape) == 3 else lip_crop
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
        if self._prev_frame is None or self._prev_frame.shape != gray.shape:
            self._prev_frame = gray
            return 0.0
        diff = np.abs(gray.astype(np.float32) - self._prev_frame.astype(np.float32))
        motion = float(diff.mean())
        self._prev_frame = gray
        return motion

    def get_lip_state(self, lip_crop):
        motion = self.optical_flow_motion(lip_crop)
        if motion < 1.0:
            state = "silent"
            confidence = 0.9
        elif motion < 6.0:
            state = "speaking"
            confidence = 0.75
        else:
            state = "fast_motion"
            confidence = 0.6
        return {
            "state": state,
            "motion_score": motion,
            "confidence": confidence,
            "is_speaking": state != "silent"
        }
