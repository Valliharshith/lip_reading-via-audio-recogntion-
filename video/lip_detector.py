# video/lip_detector.py
import cv2
import mediapipe as mp
import numpy as np
from config import LIP_PADDING

class LipDetector:
    def __init__(self):
        self._last_landmarks = None
        self._last_hw = (480, 640)
        base_options = mp.tasks.BaseOptions(
            model_asset_path=None,
            delegate=mp.tasks.BaseOptions.Delegate.CPU
        )
        # Use legacy solution - but with image writeable flag fix
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    LIPS_ALL = [61,185,40,39,37,0,267,269,270,409,
                291,375,321,405,314,17,84,181,91,146,
                78,191,80,81,82,13,312,311,310,415,
                308,324,318,402,317,14,87,178,88,95]

    def detect(self, frame):
        if frame is None:
            return None, None, frame

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()

        self._last_landmarks = None
        if not results.multi_face_landmarks:
            return None, None, annotated

        face_landmarks = results.multi_face_landmarks[0]
        lip_pts = []
        for idx in self.LIPS_ALL:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            lip_pts.append((x, y))

        lip_pts = np.array(lip_pts)
        x1 = max(0, lip_pts[:,0].min() - LIP_PADDING)
        y1 = max(0, lip_pts[:,1].min() - LIP_PADDING)
        x2 = min(w, lip_pts[:,0].max() + LIP_PADDING)
        y2 = min(h, lip_pts[:,1].max() + LIP_PADDING)

        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.putText(annotated, "LIPS", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        self._last_landmarks = face_landmarks
        lip_crop = frame[y1:y2, x1:x2]
        if lip_crop.size == 0:
            return None, None, annotated

        return lip_crop, (x1,y1,x2,y2), annotated

    def preprocess_lip(self, lip_crop, target_size=(96,48)):
        if lip_crop is None:
            return np.zeros((*target_size[::-1], 3), dtype=np.float32)
        resized = cv2.resize(lip_crop, target_size)
        return resized.astype(np.float32) / 255.0

    def release(self):
        self.face_mesh.close()
