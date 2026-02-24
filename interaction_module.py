"""
HandSense Pro — Interaction Engine
Modes: DETECT | DRAW | SIGNATURE | MOUSE | ZOOM | LASER | MULTI
"""

import cv2
import numpy as np
import base64
import math
import time
import pyautogui
from collections import deque
from typing import Dict, List, Optional, Tuple


class InteractionModule:
    """
    Advanced touchless interaction system.

    DETECT   : Pure detection overlay, no canvas interaction
    DRAW     : Air brush — index draws, pinch erases, fist lifts
    SIGNATURE: Smooth signature with pressure simulation
    MOUSE    : Two-finger cursor + pinch click + scroll gesture
    ZOOM     : Two-hand pinch-to-zoom simulation
    LASER    : Laser pointer mode (no drawing)
    MULTI    : Both hands: left = erase, right = draw
    """

    MODES = ['detect','draw','signature','mouse','zoom','laser','multi']

    COLOR_MAP = {
        'white':  (255,255,255), 'red':    (0,0,255),   'green': (0,255,0),
        'blue':   (255,100,0),   'yellow': (0,255,255),  'cyan':  (255,255,0),
        'purple': (255,0,255),   'orange': (0,165,255),  'pink':  (147,20,255),
        'lime':   (0,255,100),   'teal':   (255,200,0),  'gold':  (0,215,255),
    }

    def __init__(self):
        self.current_mode  = 'draw'
        self.canvas        = None
        self.sig_canvas    = None      # separate clean signature canvas
        self.prev_point    = None
        self.prev_right    = None
        self.prev_left     = None
        self.brush_size    = 8
        self.brush_color   = 'white'
        self.eraser_size   = 45
        self.opacity       = 1.0

        # Laser pointer
        self.laser_trail   = deque(maxlen=15)

        # Mouse control
        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth        = 6
        self.cursor        = [self.screen_w//2, self.screen_h//2]
        self._last_click   = 0
        self._last_rclick  = 0
        self._scroll_prev  = None

        # Zoom state
        self._zoom_init_dist = None
        self._zoom_level     = 1.0

        # Drawing pressure simulation (velocity-based)
        self._draw_vel     = deque(maxlen=5)

        # Stats
        self.stats = {'strokes':0, 'points':0, 'clicks':0, 'mode_time': time.time()}

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        if mode in self.MODES:
            self.current_mode = mode
            self.prev_point = self.prev_right = self.prev_left = None
            self.stats['mode_time'] = time.time()

    def clear_canvas(self):
        if self.canvas is not None:    self.canvas[:] = 0
        if self.sig_canvas is not None: self.sig_canvas[:] = 0
        self.prev_point = None
        self.stats['strokes'] = self.stats['points'] = 0

    def get_canvas_b64(self) -> str:
        target = self.sig_canvas if self.current_mode == 'signature' else self.canvas
        if target is None: return ''
        _, buf = cv2.imencode('.png', target)
        return base64.b64encode(buf).decode()

    def process(self, frame: np.ndarray, hand_data: Dict) -> Tuple[np.ndarray, Dict]:
        h, w = frame.shape[:2]
        self._ensure_canvas(h, w)

        action = {'action':'idle','cursor_pos':[0,0],'mode':self.current_mode,'stats':self.stats}
        hands  = hand_data.get('hands', [])

        if not hands:
            self.prev_point = self.prev_right = self.prev_left = None
            self._overlay(frame)
            self._draw_mode_hud(frame)
            return frame, action

        primary = hands[0]

        dispatch = {
            'detect':    self._detect_mode,
            'draw':      self._draw_mode,
            'signature': self._sig_mode,
            'mouse':     self._mouse_mode,
            'zoom':      self._zoom_mode,
            'laser':     self._laser_mode,
            'multi':     self._multi_mode,
        }
        fn = dispatch.get(self.current_mode, self._detect_mode)
        action = fn(frame, primary, hands, w, h) or action

        self._overlay(frame)
        self._draw_mode_hud(frame)
        return frame, action

    # ── Modes ─────────────────────────────────────────────────────────────────

    def _detect_mode(self, frame, hand, hands, w, h):
        # Just show detection info
        self._draw_hand_info_panel(frame, hand, w, h)
        return {'action':'detecting','cursor_pos':[hand['index_tip']['x'], hand['index_tip']['y']],'mode':'detect'}

    def _draw_mode(self, frame, hand, hands, w, h):
        fu = hand['fingers_up']
        ix, iy = hand['index_tip']['x'], hand['index_tip']['y']
        tx, ty = hand['thumb_tip']['x'], hand['thumb_tip']['y']
        pinch  = hand['pinch_dist']

        only_index = fu[1]==1 and fu[2]==0 and fu[3]==0 and fu[4]==0
        is_erase   = pinch < 40 and fu[1]==1

        color = self.COLOR_MAP.get(self.brush_color, (255,255,255))

        if is_erase:
            cv2.circle(self.canvas,(ix,iy),self.eraser_size,(0,0,0),-1)
            self._draw_eraser_cursor(frame, ix, iy)
            self.prev_point = None
            return {'action':'erasing','cursor_pos':[ix,iy],'mode':'draw'}

        if only_index:
            size = self._velocity_brush(hand['velocity']['speed'])
            if self.prev_point:
                self._smooth_line(self.canvas, self.prev_point, (ix,iy), color, size)
                self.stats['points'] += 1
            else:
                cv2.circle(self.canvas,(ix,iy),size//2,color,-1)
                self.stats['strokes'] += 1
            self.prev_point = (ix,iy)
            self._draw_brush_cursor(frame, ix, iy, color, size)
            return {'action':'drawing','cursor_pos':[ix,iy],'mode':'draw','brush_size':size}

        self.prev_point = None
        return {'action':'pen_up','cursor_pos':[ix,iy],'mode':'draw'}

    def _sig_mode(self, frame, hand, hands, w, h):
        fu = hand['fingers_up']
        ix, iy = hand['index_tip']['x'], hand['index_tip']['y']
        only_index = fu[1]==1 and sum(fu[2:5])==0

        sig_color = (0,255,150)

        if only_index:
            speed = hand['velocity']['speed']
            size  = max(2, 5 - int(speed * 0.03))  # pressure: slower = thicker
            if self.prev_point:
                # Catmull-Rom-like smoothing
                self._smooth_line(self.sig_canvas, self.prev_point, (ix,iy), sig_color, size)
                self._smooth_line(self.canvas,     self.prev_point, (ix,iy), sig_color, size)
            self.prev_point = (ix,iy)
            cv2.circle(frame,(ix,iy),4,sig_color,-1)
            return {'action':'signing','cursor_pos':[ix,iy],'mode':'signature'}

        self.prev_point = None
        return {'action':'pen_up','cursor_pos':[ix,iy],'mode':'signature'}

    def _mouse_mode(self, frame, hand, hands, w, h):
        fu    = hand['fingers_up']
        ix,iy = hand['index_tip']['x'], hand['index_tip']['y']
        tx,ty = hand['thumb_tip']['x'], hand['thumb_tip']['y']
        mx2,my2 = hand['middle_tip']['x'], hand['middle_tip']['y']

        # Map to screen
        sx = np.interp(ix, [80,w-80], [0,self.screen_w])
        sy = np.interp(iy, [80,h-80], [0,self.screen_h])

        # Smooth
        cx = self.cursor[0] + (sx - self.cursor[0]) / self.smooth
        cy = self.cursor[1] + (sy - self.cursor[1]) / self.smooth
        self.cursor = [cx, cy]

        action = 'idle'

        # Move: index + middle up
        if fu[1]==1 and fu[2]==1:
            pyautogui.moveTo(int(cx), int(cy))
            action = 'move'
            self._draw_mouse_cursor(frame, ix, iy, (0,255,170))

        # Left click: pinch
        if hand['pinch_dist'] < 35 and time.time()-self._last_click > 0.55:
            pyautogui.click()
            self._last_click = time.time()
            action = 'click'
            self.stats['clicks'] += 1
            self._draw_click_flash(frame, ix, iy)

        # Right click: pinky up + index up
        if fu[1]==1 and fu[4]==1 and fu[2]==0 and time.time()-self._last_rclick > 0.8:
            pyautogui.rightClick()
            self._last_rclick = time.time()
            action = 'right_click'

        # Scroll: fist up/down movement
        if fu==[0,0,0,0,0]:
            vy = hand['velocity']['vy']
            if abs(vy) > 4:
                pyautogui.scroll(int(-vy * 0.5))
                action = 'scroll'

        return {'action':action,'cursor_pos':[int(cx),int(cy)],'mode':'mouse'}

    def _zoom_mode(self, frame, hand, hands, w, h):
        if len(hands) < 2:
            self._zoom_init_dist = None
            cv2.putText(frame,"Show 2 hands to zoom",(20,H-40 if (H:=frame.shape[0]) else 60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,200,255),2,cv2.LINE_AA)
            return {'action':'waiting','cursor_pos':[0,0],'mode':'zoom'}

        h0, h1 = hands[0], hands[1]
        p0 = (h0['palm_center']['x'], h0['palm_center']['y'])
        p1 = (h1['palm_center']['x'], h1['palm_center']['y'])
        dist = math.hypot(p1[0]-p0[0], p1[1]-p0[1])

        if self._zoom_init_dist is None:
            self._zoom_init_dist = dist

        self._zoom_level = dist / max(self._zoom_init_dist, 1)

        # Draw zoom line
        mid = ((p0[0]+p1[0])//2, (p0[1]+p1[1])//2)
        cv2.line(frame, p0, p1, (0,255,200), 2)
        cv2.circle(frame, mid, 8, (0,255,200), -1)
        cv2.putText(frame, f"ZOOM {self._zoom_level:.2f}x",
                    (mid[0]-40, mid[1]-16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,200), 2, cv2.LINE_AA)

        return {'action':'zooming','zoom':round(self._zoom_level,2),'cursor_pos':list(mid),'mode':'zoom'}

    def _laser_mode(self, frame, hand, hands, w, h):
        ix,iy = hand['index_tip']['x'], hand['index_tip']['y']
        fu    = hand['fingers_up']
        if fu[1]==1:
            self.laser_trail.append((ix,iy))
            # Draw fading trail
            for i in range(1, len(self.laser_trail)):
                alpha = i / len(self.laser_trail)
                c = (int(0), int(0), int(255*alpha))
                cv2.line(frame, self.laser_trail[i-1], self.laser_trail[i], c, max(1,int(alpha*4)), cv2.LINE_AA)
            # Bright tip
            cv2.circle(frame,(ix,iy),8,(0,0,255),-1)
            cv2.circle(frame,(ix,iy),16,(0,0,255),1)
            overlay = frame.copy()
            cv2.circle(overlay,(ix,iy),20,(0,50,255),-1)
            cv2.addWeighted(overlay,0.3,frame,0.7,0,frame)
        return {'action':'laser','cursor_pos':[ix,iy],'mode':'laser'}

    def _multi_mode(self, frame, hand, hands, w, h):
        """Left hand = eraser, Right hand = draw."""
        color = self.COLOR_MAP.get(self.brush_color,(255,255,255))

        for h in hands:
            fu  = h['fingers_up']
            ix  = h['index_tip']['x']; iy = h['index_tip']['y']
            lbl = h['label']

            if lbl == 'Left' and fu[1]==1:
                cv2.circle(self.canvas,(ix,iy),self.eraser_size,(0,0,0),-1)
                self._draw_eraser_cursor(frame,ix,iy)

            elif lbl == 'Right' and fu[1]==1 and fu[2]==0:
                size = self._velocity_brush(h['velocity']['speed'])
                prev = self.prev_right
                if prev:
                    self._smooth_line(self.canvas,prev,(ix,iy),color,size)
                self.prev_right = (ix,iy)
                self._draw_brush_cursor(frame,ix,iy,color,size)
            else:
                if h['label']=='Right': self.prev_right=None

        return {'action':'multi','cursor_pos':[ix,iy],'mode':'multi'}

    # ── Drawing helpers ──────────────────────────────────────────────────────

    def _smooth_line(self, canvas, p1, p2, color, size):
        """Smooth line with intermediate points."""
        dist = math.hypot(p2[0]-p1[0],p2[1]-p1[1])
        steps = max(1, int(dist / 3))
        for i in range(steps+1):
            t = i/steps
            x = int(p1[0] + (p2[0]-p1[0])*t)
            y = int(p1[1] + (p2[1]-p1[1])*t)
            cv2.circle(canvas,(x,y),size//2,color,-1)

    def _velocity_brush(self, speed: float) -> int:
        """Brush size increases slightly with speed for natural feel."""
        return max(2, min(self.brush_size + int(speed*0.05), self.brush_size+4))

    def _draw_brush_cursor(self, frame, x, y, color, size):
        cv2.circle(frame,(x,y),size+3,color,2)
        cv2.circle(frame,(x,y),3,color,-1)

    def _draw_eraser_cursor(self, frame, x, y):
        cv2.circle(frame,(x,y),self.eraser_size,(120,120,120),2)
        cv2.line(frame,(x-8,y),(x+8,y),(120,120,120),1)
        cv2.line(frame,(x,y-8),(x,y+8),(120,120,120),1)

    def _draw_mouse_cursor(self, frame, x, y, color):
        cv2.circle(frame,(x,y),10,color,2)
        cv2.circle(frame,(x,y),4,color,-1)

    def _draw_click_flash(self, frame, x, y):
        overlay = frame.copy()
        cv2.circle(overlay,(x,y),28,(0,255,255),-1)
        cv2.addWeighted(overlay,0.35,frame,0.65,0,frame)
        cv2.circle(frame,(x,y),28,(0,255,255),2)

    def _draw_hand_info_panel(self, frame, hand, w, h):
        g = hand['gesture'].replace('_',' ').title()
        fc = hand['finger_count']
        sp = hand['velocity']['speed']
        lines = [
            f"Gesture: {g}",
            f"Fingers: {fc}",
            f"Speed:   {sp:.0f}px/f",
            f"Pinch:   {hand['pinch_dist']:.0f}px",
        ]
        x0, y0 = 12, 55
        cv2.rectangle(frame,(x0-6,y0-18),(x0+170,y0+len(lines)*20+4),(10,20,40),-1)
        cv2.rectangle(frame,(x0-6,y0-18),(x0+170,y0+len(lines)*20+4),(0,255,170),1)
        for i,ln in enumerate(lines):
            cv2.putText(frame,ln,(x0,y0+i*19),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,170),1,cv2.LINE_AA)

    def _draw_mode_hud(self, frame):
        h, w = frame.shape[:2]
        mode_cols = {
            'detect':'(150,150,150)','draw':'(0,200,255)','signature':'(0,255,150)',
            'mouse':'(255,100,0)','zoom':'(255,200,0)','laser':'(0,0,255)','multi':'(255,0,200)',
        }
        colors = {
            'detect':(150,150,150),'draw':(0,200,255),'signature':(0,255,150),
            'mouse':(255,100,0),'zoom':(255,200,0),'laser':(0,0,255),'multi':(255,0,200),
        }
        color = colors.get(self.current_mode,(255,255,255))
        label = f"[ {self.current_mode.upper()} ]"
        tw    = len(label)*10
        x     = w - tw - 20
        cv2.rectangle(frame,(x-6,8),(x+tw+4,36),(10,15,30),-1)
        cv2.rectangle(frame,(x-6,8),(x+tw+4,36),color,1)
        cv2.putText(frame,label,(x,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)

    def _overlay(self, frame):
        if self.canvas is not None:
            mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
            frame[mask>0] = self.canvas[mask>0]

    def _ensure_canvas(self, h, w):
        if self.canvas is None:
            self.canvas     = np.zeros((h,w,3),dtype=np.uint8)
            self.sig_canvas = np.zeros((h,w,3),dtype=np.uint8)
