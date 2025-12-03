# stream_manager.py
import threading
import time
import base64
from queue import Queue
import config

# Try to import cv2, but don't crash if missing
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class StreamWorker:
    def __init__(self, socketio, sid, detector):
        self.socketio = socketio
        self.sid = sid
        self.detector = detector
        self.queue = Queue(maxsize=5)
        self.active = True
        self.detection_enabled = False # Default to False for safety
        
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _process_loop(self):
        while self.active:
            # Get the frame from the queue
            item = self.queue.get()
            if item is None: break
            
            timestamp, raw_payload = item

            # --- FAST PATH (PASS-THROUGH) ---
            # If AI is off OR we don't have OpenCV, just send the raw string back.
            # This mimics the 'debug_server.py' exactly.
            if (not self.detection_enabled) or (not CV2_AVAILABLE):
                self.socketio.emit('annotated_frame', {'image': raw_payload}, room=config.BROADCAST_ROOM)
                continue

            # --- SLOW PATH (AI DETECTION) ---
            # Only runs if AI is ON and we have OpenCV
            if CV2_AVAILABLE:
                try:
                    # 1. Decode
                    if ',' in raw_payload:
                        b64_data = raw_payload.split(',', 1)[1]
                    else:
                        b64_data = raw_payload
                        
                    decoded = base64.b64decode(b64_data)
                    arr = np.frombuffer(decoded, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    # 2. Detect
                    final_img = self.detector.predict(img)

                    # 3. Encode back to JPEG
                    success, jpg = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
                    if success:
                        b64_out = base64.b64encode(jpg.tobytes()).decode('utf-8')
                        payload_out = 'data:image/jpeg;base64,' + b64_out
                        self.socketio.emit('annotated_frame', {'image': payload_out}, room=config.BROADCAST_ROOM)
                except Exception as e:
                    print(f"Frame processing error: {e}")

    def stop(self):
        self.active = False
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=1)

class StreamManager:
    def __init__(self, socketio, detector):
        self.socketio = socketio
        self.detector = detector
        self.workers = {} 
        self.lock = threading.Lock()

    def start_worker(self, sid):
        with self.lock:
            if sid in self.workers:
                self.workers[sid].stop()
            worker = StreamWorker(self.socketio, sid, self.detector)
            self.workers[sid] = worker
            return worker

    def stop_worker(self, sid):
        with self.lock:
            worker = self.workers.pop(sid, None)
            if worker:
                worker.stop()

    def add_frame(self, sid, data):
        with self.lock:
            worker = self.workers.get(sid)
            if worker:
                if worker.queue.full():
                    try: worker.queue.get_nowait()
                    except: pass
                worker.queue.put_nowait((time.time(), data))

    def toggle_detection(self, sid, status):
        with self.lock:
            worker = self.workers.get(sid)
            if worker:
                worker.detection_enabled = status
                print(f"Detection for {sid} set to: {status}")