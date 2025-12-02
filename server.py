import os
import base64
import argparse
import threading
import socket
from queue import Queue

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room

import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "best.pt"
CONF_THRESH = 0.25
PROCESS_FPS = 10  

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

print("Loading model:", MODEL_PATH)
# Ensure the model exists or this will crash
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'best.pt' is in the same folder!")
    exit(1)

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
            # Run YOLO inference
            results = model.predict(source=img_bgr, conf=CONF_THRESH, verbose=False)
            
            # Plot results on the frame
            annotated = results[0].plot()  
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Prediction error:", e)
            annotated_bgr = img_bgr

        # Encode back to JPEG
        success, jpg = cv2.imencode('.jpg', annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            continue
            
        b64 = base64.b64encode(jpg.tobytes()).decode('utf-8')
        payload = 'data:image/jpeg;base64,' + b64

        # Send back to the specific room (client_sid)
        socketio.emit('annotated_frame', {'image': payload}, room=client_sid)

@app.route('/')
def index():
    return "YOLO Live Server. Go to /mobile on your phone."

@app.route('/mobile')
def mobile_page():
    return render_template('mobile.html')

@app.route('/viewer')
def viewer_page():
    return render_template('viewer.html')

@socketio.on('connect')
def on_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    print('Client disconnected:', sid)
    
    with lock:
        q = client_queues.pop(sid, None)
        thr = client_threads.pop(sid, None)
    if q:
        q.put(None)  
    if thr and thr.is_alive():
        thr.join(timeout=1)

# --- NEW: Helper for Viewer to Join Stream ---
@socketio.on('join_stream')
def on_join_stream(data):
    room = data.get('room')
    if room:
        join_room(room)
        print(f"Viewer {request.sid} joined room {room}")
        emit('stream_ready', {'message': f'Joined {room}'})

@socketio.on('start_stream')
def on_start_stream(data):
    sid = request.sid
    print(f"Starting stream for {sid}")
    
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
            return
    except Exception:
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
            q.put_nowait((0, img))
        except Exception:
            pass

def get_ip_address():
    """Finds the local IP address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='host')
    parser.add_argument('--port', default=5000, type=int, help='port')
    parser.add_argument('--model', default=None, help='model path')
    args = parser.parse_args()

    if args.model:
        MODEL_PATH = args.model

    local_ip = get_ip_address()
    port = args.port

    print("-" * 50)
    print(f" SERVER STARTED SUCCESSFULLY")
    print("-" * 50)
    print(f"1. On Mobile, open this URL:\n   http://{local_ip}:{port}/mobile")
    print("-" * 50)
    print(f"2. On Laptop (Viewer), open this URL:\n   http://localhost:{port}/viewer")
    print("-" * 50)
    print("NOTE: On Mobile Chrome, ensure you enabled 'Insecure origins treated as secure'")
    print(f"      in chrome://flags and added: http://{local_ip}:{port}")
    print("-" * 50)

    socketio.run(app, host=args.host, port=port)