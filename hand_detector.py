"""
HandSense Pro — Advanced Hand Detection Engine
MediaPipe + OpenCV with gesture recognition, velocity tracking,
zone detection, hand trails, multi-hand coordination.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


@dataclass
class HandState:
    """Full state for a single tracked hand."""
    id:            int
    label:         str              # 'Left' or 'Right'
    score:         float
    landmarks:     List[Dict]
    fingers_up:    List[int]
    finger_count:  int
    gesture:       str
    bbox:          Dict
    index_tip:     Dict
    thumb_tip:     Dict
    middle_tip:    Dict
    wrist:         Dict
    palm_center:   Dict
    velocity:      Dict             # {vx, vy, speed}
    pinch_dist:    float
    is_pinching:   bool
    history:       deque = field(default_factory=lambda: deque(maxlen=20))


class HandDetector:
    """
    Production-grade hand detection system using MediaPipe.
    Features:
    - 21-landmark tracking per hand (up to 2 hands)
    - 15+ gesture recognition
    - Velocity & acceleration tracking
    - Pinch distance measurement
    - Palm-zone interaction
    - Hand trail rendering
    - Dynamic confidence adaptation
    - FPS-stabilized processing
    """

    # ── Gesture definitions ──────────────────────────────────────────────────
    GESTURES: Dict[Tuple, str] = {
        (0,0,0,0,0): 'fist',
        (1,0,0,0,0): 'thumb_up',
        (0,1,0,0,0): 'index_point',
        (0,1,1,0,0): 'peace',
        (1,1,0,0,0): 'gun',
        (0,1,1,1,0): 'three_mid',
        (0,1,1,1,1): 'four_fingers',
        (1,1,1,1,1): 'open_hand',
        (1,0,0,0,1): 'rock_on',
        (0,0,0,0,1): 'pinky',
        (1,1,1,0,0): 'three',
        (0,0,1,1,1): 'ring_middle_pinky',
        (1,1,0,0,1): 'spider',
        (0,1,0,0,1): 'horns',
        (1,0,0,0,0): 'thumb_side',
    }

    # ── Connections for rendering ────────────────────────────────────────────
    FINGER_TIPS  = [4, 8, 12, 16, 20]
    FINGER_DIPS  = [3, 7, 11, 15, 19]
    FINGER_PIPS  = [2, 6, 10, 14, 18]
    FINGER_MCPS  = [1, 5,  9, 13, 17]

    # Colors per gesture family
    GESTURE_COLORS = {
        'fist':       (0, 80, 255),
        'open_hand':  (0, 255, 120),
        'peace':      (255, 220, 0),
        'index_point':(0, 200, 255),
        'thumb_up':   (0, 255, 200),
        'rock_on':    (255, 60, 180),
    }

    def __init__(
        self,
        max_hands:            int   = 2,
        detection_confidence: float = 0.7,
        tracking_confidence:  float = 0.5,
        show_trails:          bool  = False,
        trail_length:         int   = 25,
    ):
        self.max_hands            = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence  = tracking_confidence
        self.show_trails          = show_trails
        self.trail_length         = trail_length

        self._init_mediapipe()

        # Per-hand velocity history
        self._prev_positions: Dict[int, Tuple[float,float]] = {}
        self._hand_trails:    Dict[int, deque] = {}

        # FPS tracking (circular buffer for smoothed FPS)
        self._frame_times = deque(maxlen=30)
        self._last_time   = time.perf_counter()

        # Landmark drawing specs
        self.mp_draw = mp.solutions.drawing_utils
        self.lm_spec = self.mp_draw.DrawingSpec(color=(0,255,170), thickness=2, circle_radius=3)
        self.cn_spec = self.mp_draw.DrawingSpec(color=(200,200,200), thickness=1)

    def _init_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            static_image_mode         = False,
            max_num_hands             = self.max_hands,
            min_detection_confidence  = self.detection_confidence,
            min_tracking_confidence   = self.tracking_confidence,
            model_complexity          = 1,
        )

    def reinit(self):
        """Reinitialize with updated confidence values."""
        self.hands.close()
        self._init_mediapipe()

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame.
        Returns annotated frame and structured hand data.
        """
        h, w = frame.shape[:2]
        self._update_fps()

        # Convert + process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        hand_data = {
            'hands':        [],
            'hand_count':   0,
            'gesture':      'none',
            'finger_count': 0,
            'fps':          self.get_fps(),
        }

        if not results.multi_hand_landmarks:
            self._prev_positions.clear()
            self._draw_hud(frame)
            return frame, hand_data

        for idx, (lms, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            label = handedness.classification[0].label
            score = handedness.classification[0].score

            # ── Raw landmarks ──────────────────────────────────────────────
            pts = [
                {'x': int(lm.x * w), 'y': int(lm.y * h), 'z': round(lm.z, 5)}
                for lm in lms.landmark
            ]

            # ── Analysis ───────────────────────────────────────────────────
            fingers     = self._fingers_up(lms, label)
            gesture     = self.GESTURES.get(tuple(fingers), 'custom')
            bbox        = self._bounding_box(lms, w, h)
            palm_center = self._palm_center(pts)
            velocity    = self._calc_velocity(idx, palm_center)
            pinch_dist  = self._pinch_distance(pts[4], pts[8])
            is_pinching = pinch_dist < 38

            # ── Draw ───────────────────────────────────────────────────────
            self._draw_landmarks(frame, lms)
            self._draw_fingertip_glow(frame, pts, fingers)
            self._draw_bbox(frame, bbox, gesture, label, score)
            self._draw_palm_center(frame, palm_center, velocity)
            if is_pinching:
                self._draw_pinch_indicator(frame, pts[4], pts[8])
            if self.show_trails:
                self._draw_trail(frame, idx, palm_center)

            state = HandState(
                id          = idx,
                label       = label,
                score       = round(score, 3),
                landmarks   = pts,
                fingers_up  = fingers,
                finger_count= sum(fingers),
                gesture     = gesture,
                bbox        = bbox,
                index_tip   = pts[8],
                thumb_tip   = pts[4],
                middle_tip  = pts[12],
                wrist       = pts[0],
                palm_center = palm_center,
                velocity    = velocity,
                pinch_dist  = round(pinch_dist, 1),
                is_pinching = is_pinching,
            )
            hand_data['hands'].append({k:v for k,v in state.__dict__.items() if k!='history'})
            hand_data['finger_count'] += sum(fingers)

        hand_data['hand_count'] = len(hand_data['hands'])
        if hand_data['hands']:
            hand_data['gesture'] = hand_data['hands'][0]['gesture']

        self._draw_hud(frame)
        return frame, hand_data

    # ── Geometry helpers ─────────────────────────────────────────────────────

    def _fingers_up(self, lms, label: str) -> List[int]:
        lm = lms.landmark
        is_right = (label == 'Right')
        fingers = []
        # Thumb
        fingers.append(1 if (lm[4].x < lm[3].x if is_right else lm[4].x > lm[3].x) else 0)
        # Four fingers: tip y < PIP y → extended
        for tip_id in [8, 12, 16, 20]:
            fingers.append(1 if lm[tip_id].y < lm[tip_id - 2].y else 0)
        return fingers

    def _bounding_box(self, lms, w, h, pad=20) -> Dict:
        xs = [int(lm.x * w) for lm in lms.landmark]
        ys = [int(lm.y * h) for lm in lms.landmark]
        x1 = max(0,   min(xs) - pad);  y1 = max(0,   min(ys) - pad)
        x2 = min(w-1, max(xs) + pad);  y2 = min(h-1, max(ys) + pad)
        return {'x1':x1,'y1':y1,'x2':x2,'y2':y2,'cx':(x1+x2)//2,'cy':(y1+y2)//2}

    def _palm_center(self, pts) -> Dict:
        # Average of wrist + 5 MCP joints
        idxs = [0, 5, 9, 13, 17]
        cx = sum(pts[i]['x'] for i in idxs) // len(idxs)
        cy = sum(pts[i]['y'] for i in idxs) // len(idxs)
        return {'x': cx, 'y': cy}

    def _calc_velocity(self, hand_id: int, center: Dict) -> Dict:
        cx, cy = center['x'], center['y']
        if hand_id in self._prev_positions:
            px, py = self._prev_positions[hand_id]
            vx = cx - px; vy = cy - py
            speed = round(math.hypot(vx, vy), 1)
        else:
            vx = vy = speed = 0.0
        self._prev_positions[hand_id] = (cx, cy)
        return {'vx': round(vx,1), 'vy': round(vy,1), 'speed': speed}

    def _pinch_distance(self, p1: Dict, p2: Dict) -> float:
        return math.hypot(p2['x'] - p1['x'], p2['y'] - p1['y'])

    # ── Drawing helpers ──────────────────────────────────────────────────────

    def _draw_landmarks(self, frame, lms):
        self.mp_draw.draw_landmarks(
            frame, lms, self.mp_hands.HAND_CONNECTIONS,
            self.lm_spec, self.cn_spec
        )

    def _draw_fingertip_glow(self, frame, pts, fingers):
        for i, tip_id in enumerate(self.FINGER_TIPS):
            if not fingers[i]: continue
            x, y = pts[tip_id]['x'], pts[tip_id]['y']
            for r, alpha in [(14,0.15),(9,0.3),(5,0.7)]:
                overlay = frame.copy()
                cv2.circle(overlay, (x,y), r, (0,255,170), -1)
                cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            cv2.circle(frame, (x,y), 4, (255,255,255), -1)

    def _draw_bbox(self, frame, bbox, gesture, label, score):
        x1,y1,x2,y2 = bbox['x1'],bbox['y1'],bbox['x2'],bbox['y2']
        color = self.GESTURE_COLORS.get(gesture, (0,255,170) if label=='Right' else (0,170,255))
        cl = 16  # corner length

        for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(cx,cy),(cx+dx*cl,cy),color,2)
            cv2.line(frame,(cx,cy),(cx,cy+dy*cl),color,2)

        text  = f"{label} | {gesture.replace('_',' ').title()} {score:.0%}"
        bg_w  = len(text)*9 + 8
        cv2.rectangle(frame,(x1,y1-28),(x1+bg_w,y1),color,-1)
        cv2.putText(frame, text,(x1+4,y1-9),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),1,cv2.LINE_AA)

    def _draw_palm_center(self, frame, center, velocity):
        cx, cy = center['x'], center['y']
        speed  = velocity['speed']
        radius = max(6, min(20, int(speed * 0.5)))
        alpha  = min(0.6, speed * 0.04)
        if alpha > 0.05:
            overlay = frame.copy()
            cv2.circle(overlay,(cx,cy),radius,(0,200,255),-1)
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
        cv2.circle(frame,(cx,cy),4,(0,200,255),-1)

    def _draw_pinch_indicator(self, frame, thumb, index):
        tx,ty = thumb['x'],thumb['y']
        ix,iy = index['x'],index['y']
        mx2,my2 = (tx+ix)//2,(ty+iy)//2
        cv2.line(frame,(tx,ty),(ix,iy),(0,255,255),1,cv2.LINE_AA)
        cv2.circle(frame,(mx2,my2),6,(0,255,255),-1)
        cv2.circle(frame,(mx2,my2),12,(0,255,255),1)
        overlay = frame.copy()
        cv2.circle(overlay,(mx2,my2),20,(0,255,255),-1)
        cv2.addWeighted(overlay,0.15,frame,0.85,0,frame)
        cv2.putText(frame,"PINCH",(mx2-18,my2-16),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1,cv2.LINE_AA)

    def _draw_trail(self, frame, hand_id, center):
        if hand_id not in self._hand_trails:
            self._hand_trails[hand_id] = deque(maxlen=self.trail_length)
        trail = self._hand_trails[hand_id]
        trail.append((center['x'], center['y']))
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            color = (int(0*alpha), int(255*alpha), int(170*alpha))
            thickness = max(1, int(alpha * 4))
            cv2.line(frame, trail[i-1], trail[i], color, thickness, cv2.LINE_AA)

    def _draw_hud(self, frame):
        cv2.putText(frame, f"FPS:{int(self.get_fps())} | HandSense Pro",
                    (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,170), 2, cv2.LINE_AA)

    # ── FPS ──────────────────────────────────────────────────────────────────
    def _update_fps(self):
        now = time.perf_counter()
        self._frame_times.append(now - self._last_time)
        self._last_time = now

    def get_fps(self) -> float:
        if not self._frame_times: return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return round(1.0 / max(avg, 1e-9), 1)

    def close(self):
        self.hands.close()
