 # fusion/fusion_engine.py
from config import MODE

class FusionEngine:
    def __init__(self, mode=None):
        self.mode = mode or MODE

    def fuse(self, audio_text, audio_rms, lip_state):
        if self.mode == "audio":
            return self._audio_only(audio_text, audio_rms)
        elif self.mode == "video":
            return self._video_only(lip_state)
        else:
            return self._late_fusion(audio_text, audio_rms, lip_state)

    # AUDIO MODE → Only Whisper decides
    def _audio_only(self, text, rms):
        return {
            "text": text if rms > 0.002 and text.strip() else "",
            "confidence": min(1.0, rms * 20) if text.strip() else 0.0,
            "mode": "AUDIO"
        }

    # VIDEO MODE → NEVER fabricate words
    def _video_only(self, lip_state):
        speaking = lip_state.get("is_speaking", False)
        confidence = lip_state.get("confidence", 0.0) if speaking else 0.0

        return {
            "text": "",  # IMPORTANT: never generate fake speech
            "confidence": confidence,
            "mode": "VIDEO"
        }

    # FUSION MODE → Whisper text ONLY if real audio exists
    def _late_fusion(self, text, rms, lip_state):
        speaking = lip_state.get("is_speaking", False)

        # Real speech only if Whisper heard it
        if rms > 0.002 and text.strip():
            confidence = 0.9 if speaking else 0.7
            out_text = text
        else:
            confidence = 0.0
            out_text = ""

        return {
            "text": out_text,
            "confidence": confidence,
            "mode": "FUSION"
        }