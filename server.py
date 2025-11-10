
import os
import base64
import io
import argparse
import threading
from queue import Queue

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "best.pt"
CONF_THRESH = 0.25
PROCESS_FPS = 10  


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")


print("Loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)


client_queues = {}
client_threads = {}
lock = threading.Lock()

def process_frames(client_sid, q):
    """Background worker: consume frames from q, run detection, emit annotated frame"""
    import time
    last_time = 0
    min_interval = 1.0 / PROCESS_FPS
    while True:
        item = q.get()
        if item is None:
            
            break
        timestamp, img_bgr = item
        
        elapsed = time.time() - last_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        last_time = time.time()

        
        try:
            
            results = model.predict(source=img_bgr, conf=CONF_THRESH, verbose=False)

            
            annotated = results[0].plot()  
            
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Prediction error:", e)
            
            annotated_bgr = img_bgr

        
        success, jpg = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            print("Failed to encode jpg")
            continue
        b64 = base64.b64encode(jpg.tobytes()).decode('utf-8')
        payload = 'data:image/jpeg;base64,' + b64

        
        socketio.emit('annotated_frame', {'image': payload}, room=client_sid)

@app.route('/')
def index():
    return "YOLO live server running. Open /mobile on phone and /viewer on laptop."

@app.route('/mobile')
def mobile_page():
    return render_template('mobile.html')

@app.route('/viewer')
def viewer_page():
    return render_template('viewer.html')

@socketio.on('connect')
def on_connect():
    print('Client connected', request.sid)

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    print('Client disconnected', sid)
    
    with lock:
        q = client_queues.pop(sid, None)
        thr = client_threads.pop(sid, None)
    if q:
        q.put(None)  
    if thr and thr.is_alive():
        thr.join(timeout=1)

@socketio.on('start_stream')
def on_start_stream(data):
    """Called by mobile client to register as a sender.
       data can contain optional info.
    """
    sid = request.sid
    print(f"start_stream from {sid}")
    
    q = Queue(maxsize=5)
    thr = threading.Thread(target=process_frames, args=(sid, q), daemon=True)
    with lock:
        client_queues[sid] = q
        client_threads[sid] = thr
    thr.start()
    
    join_room(sid)
    emit('stream_ready', {'message': 'server ready'})

@socketio.on('stop_stream')
def on_stop_stream():
    sid = request.sid
    print('stop_stream', sid)
    with lock:
        q = client_queues.pop(sid, None)
        thr = client_threads.pop(sid, None)
    if q:
        q.put(None)
    if thr and thr.is_alive():
        thr.join(timeout=1)
    leave_room(sid)

@socketio.on('frame')
def on_frame(data):
    """Receive base64 frame from mobile, decode and enqueue for processing."""
    sid = request.sid
    b64 = data.get('image', None)
    if b64 is None:
        return
    
    if ',' in b64:
        b64 = b64.split(',',1)[1]
    try:
        decoded = base64.b64decode(b64)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  
        if img is None:
            print("Warning: decoded image is None")
            return
    except Exception as e:
        print("Error decoding frame:", e)
        return

    
    with lock:
        q = client_queues.get(sid)
    if q:
        
        try:
            if q.full():
                try:
                    _ = q.get_nowait()
                except:
                    pass
            q.put_nowait((time.time(), img))
        except Exception as e:
            print("Queue put error:", e)
    else:
        
        print("No queue for sid; ignoring frame")

if __name__ == '__main__':
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='host')
    parser.add_argument('--port', default=5000, type=int, help='port')
    parser.add_argument('--model', default=None, help='model path (override env)')
    parser.add_argument('--fps', default=None, type=int, help='processing fps')
    parser.add_argument('--conf', default=None, type=float, help='confidence threshold')
    args = parser.parse_args()
    if args.model:
        MODEL_PATH = args.model
    if args.fps:
        PROCESS_FPS = args.fps
    if args.conf:
        CONF_THRESH = args.conf

    print("Server starting on %s:%d" % (args.host, args.port))
    socketio.run(app, host=args.host, port=args.port)
