# main.py
import cv2
import time
import sys
from video.frame_buffer import FrameBuffer
from video.lip_detector import LipDetector
from inference.realtime_engine import RealtimeEngine
from config import MODE, FRAME_WIDTH, FRAME_HEIGHT
from models.lip_reader import LipShapeReader

def draw_overlay(frame, result, fps, lip_bbox):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # FPS
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Mode badge
    mode = result.get('mode', 'FUSION')
    mode_colors = {"AUDIO": (255, 100, 0), "VIDEO": (0, 200, 255), "FUSION": (0, 255, 100)}
    cv2.putText(overlay, f"MODE: {mode}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_colors.get(mode, (200,200,200)), 2)

    # Lip motion bar
    lip_state = result.get('lip_state', {})
    motion = lip_state.get('motion_score', 0)
    bar_w = min(int(motion * 10), w - 20)
    cv2.rectangle(overlay, (10, h - 40), (10 + bar_w, h - 25), (0, 200, 255), -1)
    cv2.putText(overlay, f"LIP MOTION: {motion:.1f}  [{lip_state.get('state','?')}]",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Audio RMS bar
    rms = result.get('audio_rms', 0)
    rms_bar = min(int(rms * 500), w - 20)
    cv2.rectangle(overlay, (10, h - 60), (10 + rms_bar, h - 45), (100, 255, 100), -1)
    cv2.putText(overlay, f"AUDIO RMS: {rms:.4f}", (10, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

    # Transcription — bottom center
    text = result.get('text', '')
    conf = result.get('confidence', 0)
    if text:
        # Background box for text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = max(0, (w - text_size[0]) // 2)
        ty = h - 110
        cv2.rectangle(overlay, (tx - 8, ty - 30), (tx + text_size[0] + 8, ty + 8),
                      (0, 0, 0), -1)
        cv2.putText(overlay, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(overlay, f"conf: {conf:.2f}", (tx, ty + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1)

    # Confidence color tint on lip bbox
    if lip_bbox:
        x1, y1, x2, y2 = lip_bbox
        color = (0, int(255 * conf), int(255 * (1 - conf)))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    return overlay


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else MODE
    print(f"\n🚀 Starting Lip Reading System | Mode: {mode.upper()}")
    print("Press 'q' to quit | 'a' = audio mode | 'v' = video mode | 'f' = fusion\n")

    frame_buf = FrameBuffer()
    lip_det = LipDetector()
    lip_reader = LipShapeReader()
    engine = RealtimeEngine()

    print("[Main] Loading models...")
    engine.start()
    frame_buf.start()
    time.sleep(2)  # Let buffers fill

    fps = 0
    frame_count = 0
    fps_timer = time.time()

    cv2.namedWindow("LipRead System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LipRead System", FRAME_WIDTH, FRAME_HEIGHT)

    while True:
        frame = frame_buf.get_latest()
        if frame is None:
            time.sleep(0.01)
            continue

        # Lip detection
        lip_crop, lip_bbox, annotated_frame = lip_det.detect(frame)

        # Inference
        result = engine.process_frame(lip_crop)
        if engine.fusion.mode == "video" and hasattr(lip_det, "_last_landmarks") and lip_det._last_landmarks:
            vphrase, vconf = lip_reader.process_landmarks(lip_det._last_landmarks)
            if vphrase:
                result["text"] = vphrase
                result["confidence"] = vconf
                result["mode"] = "VIDEO"

        # FPS calc
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        # Draw overlay
        display = draw_overlay(annotated_frame, result, fps, lip_bbox)
        cv2.imshow("LipRead System", display)

        # Key controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            print("Quitting...")
            break
        elif key == ord('a'):
            engine.fusion.mode = "audio"
            print("Switched to AUDIO mode")
        elif key == ord('v'):
            engine.fusion.mode = "video"
            print("Switched to VIDEO mode")
        elif key == ord('f'):
            engine.fusion.mode = "fusion"
            print("Switched to FUSION mode")

    print("\n[Main] Shutting down...")
    frame_buf.stop()
    engine.stop()
    lip_det.release()
    cv2.destroyAllWindows()
    print("[Main] Done ✅")


if __name__ == "__main__":
    main()