# models/lip_reader.py

class LipShapeReader:
    def __init__(self):
        self._speaking_frames = 0
        self._silent_frames = 0
        print("[LipReader] Visual lip shape reader ready")

    def process_landmarks(self, landmarks):
        if landmarks is None:
            return "", 0.0
        TOP = landmarks.landmark[13]
        BOTTOM = landmarks.landmark[14]
        LEFT = landmarks.landmark[61]
        RIGHT = landmarks.landmark[291]
        mouth_h = abs(BOTTOM.y - TOP.y)
        mouth_w = abs(RIGHT.x - LEFT.x)
        mar = mouth_h / (mouth_w + 1e-6)

        if mar > 0.035:
            self._speaking_frames += 1
            self._silent_frames = 0
            if self._speaking_frames >= 3:
                return "[Wide open]", 0.9
        elif mar > 0.020:
            self._speaking_frames += 1
            self._silent_frames = 0
            if self._speaking_frames >= 3:
                return "[Mouth open]", 0.75
        elif mar > 0.010:
            self._speaking_frames += 1
            self._silent_frames = 0
            if self._speaking_frames >= 3:
                return "[Lips moving]", 0.6
        else:
            self._speaking_frames = 0
            self._silent_frames += 1
            if self._silent_frames >= 5:
                return "", 0.0
        return "", 0.0
