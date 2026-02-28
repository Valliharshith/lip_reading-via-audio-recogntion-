import os, sys
try:
    import imageio_ffmpeg
    os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
except: pass
from fastapi import WebSocket, WebSocketDisconnect, FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import numpy as np, base64, cv2, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video.lip_detector import LipDetector
from models.whisper_model import WhisperASR
from models.lip_motion_model import LipMotionDetector
from models.lip_reader import LipShapeReader

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading models...")
ld = LipDetector()
asr = WhisperASR()
lm = LipMotionDetector()
lr = LipShapeReader()
print("All models ready.")

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict/frame")
async def predict_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "bad frame"})
    lip_crop, bbox, _ = ld.detect(frame)
    lip_state = lm.get_lip_state(lip_crop)
    return JSONResponse({
        "lip_state": {k: (float(v) if hasattr(v, "item") else v) for k,v in lip_state.items()} if lip_state else {},
        "bbox": [int(x) for x in bbox] if bbox else None
    })

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    import librosa
    contents = await file.read()
    ext = file.filename.split(".")[-1] if file.filename else "webm"
    with tempfile.NamedTemporaryFile(suffix="."+ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        audio, _ = librosa.load(tmp_path, sr=16000)
        os.unlink(tmp_path)
        rms = float(np.sqrt(np.mean(audio**2)))
        print(f"[Audio] RMS={rms:.4f} samples={len(audio)}")
        text = ""
        if rms > 0.02 and len(audio) >= 4000:
            text = asr.transcribe(audio)
            print(f"[Transcription] {repr(text)}")
            if text == getattr(asr, "_last_text", ""):
                text = ""
            else:
                asr._last_text = text
        return JSONResponse({
            "transcription": text,
            "rms": rms,
            "duration": len(audio)/16000,
            "speech_detected": rms > 0.002
        })
    except Exception as e:
        print(f"[AudioErr] {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return JSONResponse({"transcription": "", "rms": 0, "duration": 0, "speech_detected": False})

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    lip_crop, bbox, annotated = ld.detect(frame)
    lip_state = lm.get_lip_state(lip_crop)
    _, buf = cv2.imencode(".jpg", annotated)
    return JSONResponse({
        "lip_detected": bbox is not None,
        "bbox": [int(x) for x in bbox] if bbox else None,
        "lip_state": {k: (float(v) if hasattr(v, "item") else v) for k,v in lip_state.items()} if lip_state else {},
        "annotated_image": base64.b64encode(buf).decode()
    })
from fastapi import WebSocket, WebSocketDisconnect
import json

@app.websocket('/ws/realtime')
async def ws_realtime(websocket: WebSocket):
    await websocket.accept()
    print('[WS] Client connected')
    tbuf = ''
    last_sent_text = ''
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            result = {'text': '', 'lip_state': {}, 'confidence': 0.0, 'mode': 'FUSION', 'rms': 0.0}
            if 'frame' in payload:
                try:
                    import numpy as np, base64, cv2
                    fb = base64.b64decode(payload['frame'])
                    nparr = np.frombuffer(fb, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        lip_crop, bbox, _ = ld.detect(frame)
                        lip_state = lm.get_lip_state(lip_crop)
                        result['lip_state'] = {k:(float(v) if hasattr(v,'item') else v) for k,v in lip_state.items()} if lip_state else {}
                        result['bbox'] = [int(x) for x in bbox] if bbox else None
                        if payload.get('mode') == 'video' and hasattr(ld, '_last_landmarks') and ld._last_landmarks:
                            vphrase, vconf = lr.process_landmarks(ld._last_landmarks)
                            if vphrase:
                                result['text'] = vphrase
                                result['confidence'] = vconf
                except Exception as ve:
                    print('[VideoErr]', ve)
            if payload.get('audio'):
                try:
                    import tempfile, os
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    raw = base64.b64decode(payload['audio'])
                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
                        tmp.write(raw)
                        tmp_path = tmp.name
                    wav_path = tmp_path + ".wav"
                    import subprocess, time
                    r = subprocess.run([ffmpeg_exe, "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path], capture_output=True, timeout=10)
                    time.sleep(0.2)
                    import soundfile as sf
                    if r.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) < 100:
                        raise Exception('ffmpeg failed')
                    audio, _ = sf.read(wav_path)
                    audio = audio.astype(np.float32)
                    try: os.unlink(tmp_path)
                    except: pass
                    try: os.unlink(wav_path)
                    except: pass
                    rms = float(np.sqrt(np.mean(audio**2)))
                    result['rms'] = rms
                    print(f'[Audio] RMS={rms:.4f} samples={len(audio)}')
                    if rms > 0.02 and len(audio) >= 4000:
                        text = asr.transcribe(audio)
                        print(f'[Transcription] {repr(text)}')
                        if text.strip() and text.strip() != last_sent_text:
                            tbuf = text
                            last_sent_text = text.strip()
                            result['text'] = text
                except Exception as ae:
                    print('[AudioErr]', type(ae).__name__, str(ae))
            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print('[WS] Disconnected')
    except Exception as e:
        print('[WSErr]', e)
