"""
Virtual Air Canvas - SMOOTH DRAWING VERSION
Draw with your finger with ultra-smooth strokes!

CONTROLS:
- Move your INDEX FINGER to draw (when started)
- PINCH (thumb + index together) to select colors/tools
- FIST to stop drawing
- Press ENTER to START/STOP drawing mode
- Press C to clear canvas
- Press S to save
- Press E to toggle ERASER mode
- Press B to cycle BRUSH STYLES
- Press Z to UNDO
- Press Y to REDO
- Press +/- to adjust BRUSH size
- Press [ / ] to adjust ERASER size
- Press Q to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime
import os
from collections import deque
import random

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variables
drawing_mode_active = False
brush_thickness = 5
eraser_thickness = 15
current_color = (0, 0, 255)  # Red
prev_x, prev_y = None, None
eraser_mode = False
current_brush_style = 'MARKER'

# Smoothing buffer for coordinates
coord_buffer = deque(maxlen=5)  # Smooth last 5 points

# History for undo/redo
canvas_history = deque(maxlen=20)
redo_stack = deque(maxlen=20)

# Track if we've started a new stroke
stroke_started = False

# Colors
COLORS = {
    'RED': (0, 0, 255),
    'GREEN': (0, 255, 0),
    'BLUE': (255, 0, 0),
    'YELLOW': (0, 255, 255),
    'PINK': (255, 0, 255),
    'ORANGE': (0, 165, 255),
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0)
}

BRUSH_STYLES = ['MARKER', 'SPRAY', 'CALLIGRAPHY', 'GLOW']

class ColorButton:
    def __init__(self, x, y, w, h, name, color):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.name = name
        self.color = color
        self.selected = False
        
    def draw(self, img):
        color = (255, 255, 255) if self.selected else self.color
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), color, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), (255, 255, 255), 2)
        
        font_scale = 0.35
        text_size = cv2.getTextSize(self.name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        text_color = (0, 0, 0) if self.selected else (255, 255, 255)
        cv2.putText(img, self.name, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
    
    def is_inside(self, x, y):
        return self.x < x < self.x + self.w and self.y < y < self.y + self.h


def smooth_coordinates(x, y, buffer):
    """Apply moving average smoothing to coordinates"""
    buffer.append((x, y))
    if len(buffer) < 2:
        return x, y
    
    # Weighted average - more weight to recent points
    weights = [1, 2, 3, 4, 5]  # Most recent gets highest weight
    total_weight = sum(weights[-len(buffer):])
    
    smooth_x = sum(pt[0] * weights[i] for i, pt in enumerate(buffer)) / total_weight
    smooth_y = sum(pt[1] * weights[i] for i, pt in enumerate(buffer)) / total_weight
    
    return int(smooth_x), int(smooth_y)


def get_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def is_hand_closed(landmarks, w, h):
    """Check if hand is in a FIST (all fingers down)"""
    fingertips = [4, 8, 12, 16, 20]
    palm_base = landmarks.landmark[0]
    
    closed_count = 0
    for tip_id in fingertips:
        tip = landmarks.landmark[tip_id]
        dist = get_distance((tip.x*w, tip.y*h), (palm_base.x*w, palm_base.y*h))
        if dist < 100:
            closed_count += 1
    
    return closed_count >= 4


def is_pinching(landmarks, w, h):
    """Check if thumb and index are pinching"""
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    
    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
    
    distance = get_distance((thumb_x, thumb_y), (index_x, index_y))
    return distance < 30, (index_x, index_y)


def save_state(canvas, canvas_history, redo_stack):
    """Save current canvas state for undo"""
    canvas_history.append(canvas.copy())
    redo_stack.clear()


def interpolate_points(p1, p2, num_points=5):
    """Generate intermediate points between p1 and p2 for ultra-smooth lines"""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        x = int(p1[0] + t * (p2[0] - p1[0]))
        y = int(p1[1] + t * (p2[1] - p1[1]))
        points.append((x, y))
    return points


def draw_marker(canvas, p1, p2, color, thickness):
    """Standard smooth marker with interpolation"""
    # Add intermediate points for smoother curves
    points = interpolate_points(p1, p2, num_points=3)
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i+1], color, thickness, cv2.LINE_AA)


def draw_spray(canvas, p1, p2, color, thickness):
    """Spray paint effect - interpolated"""
    spray_radius = thickness * 3
    num_particles = int(thickness * 1.5)
    
    # Spray along the line path
    points = interpolate_points(p1, p2, num_points=3)
    for point in points:
        for _ in range(num_particles):
            offset_x = random.randint(-spray_radius, spray_radius)
            offset_y = random.randint(-spray_radius, spray_radius)
            
            if math.sqrt(offset_x**2 + offset_y**2) <= spray_radius:
                px = point[0] + offset_x
                py = point[1] + offset_y
                if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                    cv2.circle(canvas, (px, py), 1, color, -1)


def draw_calligraphy(canvas, p1, p2, color, thickness):
    """Calligraphy brush - varies thickness based on direction"""
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    varied_thickness = int(thickness * (1 + 0.5 * abs(math.sin(angle * 2))))
    
    # Smooth interpolation
    points = interpolate_points(p1, p2, num_points=3)
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i+1], color, varied_thickness, cv2.LINE_AA)


def draw_glow(canvas, p1, p2, color, thickness):
    """Glowing neon effect with smooth interpolation"""
    points = interpolate_points(p1, p2, num_points=5)
    
    for i in range(len(points) - 1):
        # Draw outer glow layers
        for j in range(3, 0, -1):
            glow_color = tuple(int(c * (0.3 + 0.2 * j)) for c in color)
            cv2.line(canvas, points[i], points[i+1], glow_color, thickness + j*2, cv2.LINE_AA)
        # Draw bright core
        cv2.line(canvas, points[i], points[i+1], color, thickness, cv2.LINE_AA)


def draw_with_style(canvas, p1, p2, color, thickness, style):
    """Apply the selected brush style"""
    if style == 'MARKER':
        draw_marker(canvas, p1, p2, color, thickness)
    elif style == 'SPRAY':
        draw_spray(canvas, p1, p2, color, thickness)
    elif style == 'CALLIGRAPHY':
        draw_calligraphy(canvas, p1, p2, color, thickness)
    elif style == 'GLOW':
        draw_glow(canvas, p1, p2, color, thickness)


def main():
    global drawing_mode_active, brush_thickness, eraser_thickness, current_color, prev_x, prev_y
    global eraser_mode, current_brush_style, canvas_history, redo_stack, coord_buffer, stroke_started
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Cannot access camera!")
        return
    
    canvas = None
    
    # Create color buttons
    buttons = []
    colors_list = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'PINK', 'ORANGE', 'WHITE', 'BLACK']
    for i, color_name in enumerate(colors_list):
        btn = ColorButton(530, 60 + i*45, 70, 35, color_name, COLORS[color_name])
        if i == 0:
            btn.selected = True
        buttons.append(btn)
    
    # Tool buttons
    eraser_btn = ColorButton(530, 60 + len(colors_list)*45 + 10, 70, 35, 'ERASER', (200, 200, 200))
    buttons.append(eraser_btn)
    
    clear_btn = ColorButton(530, 60 + len(colors_list)*45 + 55, 70, 35, 'CLEAR', (100, 100, 100))
    buttons.append(clear_btn)
    
    print("="*60)
    print("🎨 SMOOTH AIR CANVAS")
    print("="*60)
    print("⚠️  IMPORTANT: Press ENTER to START drawing mode!")
    print("\nGESTURES (when started):")
    print("  ✋ OPEN HAND → Draw with index finger")
    print("  ✊ FIST → Stop drawing")
    print("  🤏 PINCH → Select color/tool")
    print("\nKEYBOARD CONTROLS:")
    print("  ENTER → START/STOP drawing mode ⭐")
    print("  C → Clear canvas")
    print("  S → Save drawing")
    print("  E → Toggle ERASER")
    print("  B → Cycle BRUSH STYLES")
    print("  Z → UNDO")
    print("  Y → REDO")
    print("  +/- → Brush size (±2)")
    print("  [ / ] → Eraser size (±3)")
    print("  Q → Quit")
    print("="*60)
    
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        
        pinch_cooldown = 0
        is_drawing = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            if canvas is None:
                canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
                save_state(canvas, canvas_history, redo_stack)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw UI
            for btn in buttons:
                btn.draw(frame)
            
            # Status bar
            mode = "🧹 ERASER" if eraser_mode else f"🖌️ {current_brush_style}"
            
            if not drawing_mode_active:
                cv2.rectangle(frame, (0, 0), (520, 50), (0, 0, 100), -1)
                cv2.putText(frame, "⏸️ STOPPED - Press ENTER to START", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                status = "✏️ DRAWING" if is_drawing else "🖊️ READY"
                status_color = (0, 255, 0) if is_drawing else (255, 255, 0)
                
                cv2.rectangle(frame, (0, 0), (520, 50), (0, 0, 0), -1)
                cv2.putText(frame, f"{status} | {mode}", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Size indicators
            current_size = eraser_thickness if eraser_mode else brush_thickness
            size_label = "Eraser" if eraser_mode else "Brush"
            cv2.rectangle(frame, (75, 60), (200, 110), (40, 40, 40), -1)
            cv2.putText(frame, f"{size_label}: {current_size}", (80, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Color/Eraser indicator
            indicator_color = (200, 200, 200) if eraser_mode else current_color
            cv2.rectangle(frame, (10, 60), (60, 110), indicator_color, -1)
            cv2.rectangle(frame, (10, 60), (60, 110), (255, 255, 255), 2)
            if eraser_mode:
                cv2.putText(frame, "E", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            # Only process hand gestures if drawing mode is active
            if drawing_mode_active and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    index_tip = hand_landmarks.landmark[8]
                    raw_x, raw_y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Apply smoothing
                    ix, iy = smooth_coordinates(raw_x, raw_y, coord_buffer)
                    
                    hand_closed = is_hand_closed(hand_landmarks, w, h)
                    pinching, pinch_pos = is_pinching(hand_landmarks, w, h)
                    
                    # FIST = Stop drawing
                    if hand_closed:
                        if is_drawing:
                            # Save state when stroke ends
                            save_state(canvas, canvas_history, redo_stack)
                        is_drawing = False
                        prev_x, prev_y = None, None
                        coord_buffer.clear()
                        stroke_started = False
                        cv2.putText(frame, "✊ FIST - Stopped", (10, h-20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # PINCH = Select color/tool
                    elif pinching and pinch_cooldown == 0:
                        for btn in buttons:
                            if btn.is_inside(pinch_pos[0], pinch_pos[1]):
                                if btn.name == 'CLEAR':
                                    save_state(canvas, canvas_history, redo_stack)
                                    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
                                    print("🗑️ Canvas cleared!")
                                elif btn.name == 'ERASER':
                                    eraser_mode = not eraser_mode
                                    eraser_btn.selected = eraser_mode
                                    print(f"🧹 Eraser {'ON' if eraser_mode else 'OFF'} (Size: {eraser_thickness})")
                                else:
                                    current_color = COLORS[btn.name]
                                    eraser_mode = False
                                    eraser_btn.selected = False
                                    for b in buttons:
                                        if b.name != 'ERASER' and b.name != 'CLEAR':
                                            b.selected = False
                                    btn.selected = True
                                    print(f"🎨 Color: {btn.name}")
                                pinch_cooldown = 20
                                break
                        if is_drawing:
                            save_state(canvas, canvas_history, redo_stack)
                        is_drawing = False
                        prev_x, prev_y = None, None
                        coord_buffer.clear()
                        stroke_started = False
                    
                    # OPEN HAND = Draw
                    else:
                        # Start new stroke
                        if not stroke_started:
                            stroke_started = True
                        
                        is_drawing = True
                        cursor_color = (200, 200, 200) if eraser_mode else (0, 255, 0)
                        cursor_size = eraser_thickness if eraser_mode else 10
                        cv2.circle(frame, (ix, iy), cursor_size, cursor_color, -1)
                        cv2.circle(frame, (ix, iy), cursor_size, (255, 255, 255), 2)
                        
                        if prev_x is not None and prev_y is not None:
                            # Only draw if movement is not too large (prevents jumps)
                            distance = get_distance((ix, iy), (prev_x, prev_y))
                            if distance < 100:  # Prevent large jumps
                                if eraser_mode:
                                    # Smooth eraser with interpolation
                                    points = interpolate_points((prev_x, prev_y), (ix, iy), num_points=5)
                                    for point in points:
                                        cv2.circle(canvas, point, eraser_thickness, (255, 255, 255), -1)
                                else:
                                    # Apply brush style with smooth interpolation
                                    draw_with_style(canvas, (prev_x, prev_y), (ix, iy), 
                                                  current_color, brush_thickness, current_brush_style)
                        
                        prev_x, prev_y = ix, iy
                        draw_text = "🧹 ERASING!" if eraser_mode else "✏️ DRAWING!"
                        cv2.putText(frame, draw_text, (10, h-20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                if is_drawing:
                    # Save state when hand leaves frame
                    save_state(canvas, canvas_history, redo_stack)
                is_drawing = False
                prev_x, prev_y = None, None
                coord_buffer.clear()
                stroke_started = False
            
            if pinch_cooldown > 0:
                pinch_cooldown -= 1
            
            # Instructions
            if not drawing_mode_active:
                instructions = [
                    "⭐ ENTER = START",
                    "Get ready to draw!"
                ]
            else:
                instructions = [
                    "OPEN = Draw",
                    "FIST = Stop",
                    "PINCH = Select",
                    "ENTER = STOP mode"
                ]
            for i, txt in enumerate(instructions):
                cv2.putText(frame, txt, (10, h - 140 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Keyboard shortcuts
            shortcuts = [
                "E=Eraser B=Brush",
                "Z=Undo Y=Redo",
                "+/- = Brush size",
                "[/] = Eraser size"
            ]
            for i, txt in enumerate(shortcuts):
                cv2.putText(frame, txt, (220, h - 140 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Combine views
            combined = np.hstack([
                cv2.resize(frame, (640, 480)),
                np.ones((480, 20, 3), dtype=np.uint8) * 255,
                cv2.resize(canvas, (640, 480))
            ])
            
            # Labels
            cv2.putText(combined, "CAMERA", (50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined, "SMOOTH DRAWING", (660, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            cv2.imshow('Smooth Air Canvas - Ultra Smooth Drawing!', combined)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == 13:  # ENTER key
                drawing_mode_active = not drawing_mode_active
                prev_x, prev_y = None, None
                is_drawing = False
                coord_buffer.clear()
                stroke_started = False
                status_text = "STARTED ✅" if drawing_mode_active else "STOPPED ⏸️"
                print(f"\n{'='*50}")
                print(f"🎨 Drawing Mode: {status_text}")
                print(f"{'='*50}")
            
            elif key == ord('c'):
                save_state(canvas, canvas_history, redo_stack)
                canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
                print("🗑️ Canvas cleared!")
            
            elif key == ord('e'):
                eraser_mode = not eraser_mode
                eraser_btn.selected = eraser_mode
                print(f"🧹 Eraser {'ON' if eraser_mode else 'OFF'} (Size: {eraser_thickness})")
            
            elif key == ord('b'):
                current_brush_style = BRUSH_STYLES[(BRUSH_STYLES.index(current_brush_style) + 1) % len(BRUSH_STYLES)]
                print(f"🖌️ Brush Style: {current_brush_style}")
            
            elif key == ord('z'):
                if len(canvas_history) > 1:
                    redo_stack.append(canvas.copy())
                    canvas_history.pop()
                    canvas = canvas_history[-1].copy()
                    print("↶ UNDO")
            
            elif key == ord('y'):
                if redo_stack:
                    canvas_history.append(canvas.copy())
                    canvas = redo_stack.pop()
                    print("↷ REDO")
            
            elif key == ord('+') or key == ord('='):
                brush_thickness = min(brush_thickness + 2, 50)
                print(f"🖌️ Brush Size: {brush_thickness}")
            
            elif key == ord('-') or key == ord('_'):
                brush_thickness = max(brush_thickness - 2, 1)
                print(f"🖌️ Brush Size: {brush_thickness}")
            
            elif key == ord('['):
                eraser_thickness = max(eraser_thickness - 3, 3)
                print(f"🧹 Eraser Size: {eraser_thickness}")
            
            elif key == ord(']'):
                eraser_thickness = min(eraser_thickness + 3, 100)
                print(f"🧹 Eraser Size: {eraser_thickness}")
            
            elif key == ord('s'):
                if not os.path.exists('drawings'):
                    os.makedirs('drawings')
                filename = f"drawings/canvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, canvas)
                print(f"💾 Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✨ Thanks for using Smooth Air Canvas!")


if __name__ == "__main__":
    main()