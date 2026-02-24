"""
HandSense Pro — Flask Application
Real-Time Hand Detection & Interaction System
"""

import cv2
import numpy as np
import threading
import time
import base64
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from hand_detector import HandDetector
from interaction_module import InteractionModule

app = Flask(__name__)
app.config['SECRET_KEY'] = 'handsense_pro_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False)

# ─── Global state ─────────────────────────────────────────────────────────────
detector   = HandDetector(max_hands=2, detection_confidence=0.7, tracking_confidence=0.5)
interact   = InteractionModule()
camera     = None
streaming  = False
cam_lock   = threading.Lock()

# ─── Camera ───────────────────────────────────────────────────────────────────

def get_camera(width=1280, height=720):
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS,          30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    return camera

def release_camera():
    global camera
    if camera:
        camera.release()
        camera = None

# ─── Streaming generator ──────────────────────────────────────────────────────

def generate_frames():
    global streaming
    cap = get_camera()

    while streaming:
        with cam_lock:
            success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        # Process
        frame, hand_data = detector.detect(frame)
        frame, action    = interact.process(frame, hand_data)

        # Encode
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ret:
            continue

        # Emit telemetry
        if hand_data.get('hands'):
            socketio.emit('hand_data', {
                'hands':        hand_data.get('hands', []),
                'hand_count':   hand_data.get('hand_count', 0),
                'gesture':      hand_data.get('gesture', 'none'),
                'finger_count': hand_data.get('finger_count', 0),
                'fps':          detector.get_fps(),
                'action':       action.get('action', 'idle'),
                'mode':         action.get('mode', 'detect'),
                'cursor_pos':   action.get('cursor_pos', [0,0]),
                'stats':        interact.stats,
            })

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global streaming
    streaming = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start', methods=['POST'])
def start():
    global streaming
    streaming = True
    return jsonify({'ok': True, 'status': 'streaming'})

@app.route('/api/stop', methods=['POST'])
def stop():
    global streaming
    streaming = False
    release_camera()
    return jsonify({'ok': True, 'status': 'stopped'})

@app.route('/api/mode', methods=['POST'])
def set_mode():
    mode = request.json.get('mode', 'detect')
    interact.set_mode(mode)
    return jsonify({'ok': True, 'mode': mode})

@app.route('/api/canvas/clear', methods=['POST'])
def clear_canvas():
    interact.clear_canvas()
    return jsonify({'ok': True})

@app.route('/api/canvas/save', methods=['POST'])
def save_canvas():
    data = interact.get_canvas_b64()
    return jsonify({'ok': True, 'image': data, 'format': 'png'})

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify({
        'detection_confidence': detector.detection_confidence,
        'tracking_confidence':  detector.tracking_confidence,
        'max_hands':            detector.max_hands,
        'show_trails':          detector.show_trails,
        'mode':                 interact.current_mode,
        'brush_size':           interact.brush_size,
        'brush_color':          interact.brush_color,
        'eraser_size':          interact.eraser_size,
        'opacity':              interact.opacity,
    })

@app.route('/api/settings', methods=['POST'])
def update_settings():
    d = request.json or {}
    changed = []

    if 'detection_confidence' in d:
        detector.detection_confidence = float(d['detection_confidence'])
        changed.append('detection_confidence')
    if 'tracking_confidence' in d:
        detector.tracking_confidence = float(d['tracking_confidence'])
        changed.append('tracking_confidence')
    if 'max_hands' in d:
        detector.max_hands = int(d['max_hands'])
        changed.append('max_hands')
    if 'show_trails' in d:
        detector.show_trails = bool(d['show_trails'])
        changed.append('show_trails')
    if 'brush_size' in d:
        interact.brush_size = max(1, min(40, int(d['brush_size'])))
    if 'brush_color' in d:
        interact.brush_color = str(d['brush_color'])
    if 'eraser_size' in d:
        interact.eraser_size = max(10, min(80, int(d['eraser_size'])))
    if 'opacity' in d:
        interact.opacity = max(0.1, min(1.0, float(d['opacity'])))

    # Reinit mediapipe if confidence changed
    if changed:
        detector.reinit()

    return jsonify({'ok': True, 'changed': changed})

@app.route('/api/snapshot', methods=['POST'])
def snapshot():
    """Capture a single annotated frame as base64."""
    cap = get_camera()
    success, frame = cap.read()
    if not success:
        return jsonify({'ok': False, 'error': 'Camera not available'})
    frame = cv2.flip(frame, 1)
    frame, hd = detector.detect(frame)
    frame, _  = interact.process(frame, hd)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jsonify({'ok': True, 'image': base64.b64encode(buf).decode()})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'ok':           True,
        'fps':          detector.get_fps(),
        'mode':         interact.current_mode,
        'strokes':      interact.stats['strokes'],
        'points':       interact.stats['points'],
        'clicks':       interact.stats['clicks'],
    })

# ─── WebSocket ────────────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    emit('connected', {'message': 'HandSense Pro connected', 'version': '2.0'})

@socketio.on('ping')
def on_ping():
    emit('pong', {'t': time.time()})

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "═"*50)
    print("  🖐️  HandSense Pro v2.0")
    print("  ▶  http://127.0.0.1:5000")
    print("═"*50 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
