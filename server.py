# server.py
import argparse
from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room, leave_room, emit

import config
from detector import YOLODetector
from stream_manager import StreamManager

app = Flask(__name__)
# Use 'threading' mode because we know it works on your laptop
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=config.ASYNC_MODE)

# Initialize Modules
detector = YOLODetector()
stream_manager = StreamManager(socketio, detector)
current_streamer_sid = None

@app.route('/')
def index():
    return "Modular Server Running. <br><a href='/mobile'>Mobile</a> <br><a href='/viewer'>Viewer</a>"

@app.route('/mobile')
def mobile_page(): return render_template('mobile.html')

@app.route('/viewer')
def viewer_page(): return render_template('viewer.html')

@socketio.on('connect')
def on_connect():
    # Viewers automatically join the broadcast room
    join_room(config.BROADCAST_ROOM)
    print(f"Client Connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    global current_streamer_sid
    if request.sid == current_streamer_sid:
        stream_manager.stop_worker(request.sid)
        current_streamer_sid = None
    print(f"Client Disconnected: {request.sid}")

@socketio.on('start_stream')
def on_start_stream(data):
    global current_streamer_sid
    sid = request.sid
    current_streamer_sid = sid
    
    join_room(config.BROADCAST_ROOM)
    stream_manager.start_worker(sid)
    
    print(f"Stream STARTED by {sid}")
    socketio.emit('stream_ready', {'message': 'LIVE'}, room=config.BROADCAST_ROOM)

@socketio.on('stop_stream')
def on_stop_stream():
    sid = request.sid
    stream_manager.stop_worker(sid)
    leave_room(sid)

@socketio.on('toggle_detection')
def on_toggle_detection(data):
    enabled = data.get('enabled')
    if current_streamer_sid:
        stream_manager.toggle_detection(current_streamer_sid, enabled)

@socketio.on('frame')
def on_frame(data):
    sid = request.sid
    # Pass the RAW data directly to the manager
    image_data = data.get('image')
    if image_data:
        stream_manager.add_frame(sid, image_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()

    print(f"--- MODULAR SERVER RUNNING on Port {args.port} ---")
    socketio.run(app, host='0.0.0.0', port=args.port)