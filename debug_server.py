# debug_server.py
import argparse
from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room, leave_room, emit

# Setup Flask and SocketIO
app = Flask(__name__)
# We use 'threading' to avoid eventlet issues for this simple test
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading") 

# Global room name
BROADCAST_ROOM = 'live_stream'

@app.route('/')
def index():
    return "Debug Server. <br><a href='/mobile'>Mobile</a> <br><a href='/viewer'>Viewer</a>"

@app.route('/mobile')
def mobile_page():
    return render_template('mobile.html')

@app.route('/viewer')
def viewer_page():
    return render_template('viewer.html')

@socketio.on('connect')
def on_connect():
    print(f"Client Connected: {request.sid}")
    # Everyone joins the same room immediately
    join_room(BROADCAST_ROOM)

@socketio.on('start_stream')
def on_start_stream(data):
    print(f"Stream STARTED by: {request.sid}")
    emit('stream_ready', {'message': 'LIVE (Debug Mode)'}, room=BROADCAST_ROOM)

@socketio.on('frame')
def on_frame(data):
    # DEBUG: Print the first received frame size to confirm data is arriving
    image_data = data.get('image')
    
    if image_data:
        # Just echo it back immediately. No processing.
        emit('annotated_frame', {'image': image_data}, room=BROADCAST_ROOM)
    else:
        print("Warning: Received empty frame!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()

    print(f"--- DEBUG SERVER RUNNING on Port {args.port} ---")
    socketio.run(app, host='0.0.0.0', port=args.port)