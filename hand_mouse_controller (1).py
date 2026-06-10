"""
Hand Mouse Controller - Compatible with MediaPipe 0.10.30+
Uses the new MediaPipe Tasks API
"""
import cv2
import numpy as np
import pyautogui
import time
import os
import urllib.request
from collections import deque

# ========== MEDIAPIPE IMPORTS ==========
try:
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("❌ MediaPipe not installed!")
    print("Run: pip install mediapipe")
    exit(1)

# ========== HAND LANDMARK INDICES ==========
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# ========== HAND CONNECTIONS FOR DRAWING ==========
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (5, 6), (6, 7), (7, 8),                     # Index finger
    (9, 10), (10, 11), (11, 12),                # Middle finger
    (13, 14), (14, 15), (15, 16),               # Ring finger
    (17, 18), (18, 19), (19, 20),               # Pinky
    (0, 5), (5, 9), (9, 13), (13, 17),         # Palm
    (0, 17)                                     # Palm base
]


class HandMouseController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")

        # Smoothing
        self.position_buffer = deque(maxlen=7)

        # Click handling
        self.click_cooldown = 0.25
        self.last_click_time = 0
        self.is_pinching = False
        self.pinch_threshold = 0.06

        # Movement
        self.sensitivity = 1.8
        self.frame_width = None
        self.frame_height = None

        # Initialize MediaPipe Hands
        self._setup_mediapipe()

    def _setup_mediapipe(self):
        """Setup MediaPipe Hands with proper API"""
        # Try new API first (0.10.30+)
        try:
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
            from mediapipe.tasks.python.core import BaseOptions

            model_path = self._download_model()
            base_options = BaseOptions(model_asset_path=model_path)
            options = HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = HandLandmarker.create_from_options(options)
            self.using_new_api = True
            print("✅ Using MediaPipe Tasks API (new)")
            return
        except Exception as e:
            print(f"New API failed: {e}")

        # Fallback to old API
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.using_new_api = False
            print("✅ Using MediaPipe Solutions API (old)")
        except Exception as e:
            print(f"Old API also failed: {e}")
            print("Please reinstall: pip install --upgrade mediapipe")
            exit(1)

    def _download_model(self):
        """Download hand landmarker model if not present"""
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

        if os.path.exists(model_path):
            return model_path

        print("📥 Downloading hand landmarker model (10MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

        try:
            urllib.request.urlretrieve(url, model_path)
            print("✅ Model downloaded successfully!")
            return model_path
        except Exception as e:
            print(f"❌ Failed to download: {e}")
            print("Please download manually from:")
            print(url)
            print(f"And save it as: {model_path}")
            exit(1)

    def smooth_position(self, x, y):
        """Smooth cursor movement"""
        self.position_buffer.append((x, y))
        if len(self.position_buffer) < 3:
            return x, y

        # Weighted average (more weight to recent positions)
        weights = [i + 1 for i in range(len(self.position_buffer))]
        total_weight = sum(weights)

        avg_x = sum(p[0] * w for p, w in zip(self.position_buffer, weights)) / total_weight
        avg_y = sum(p[1] * w for p, w in zip(self.position_buffer, weights)) / total_weight

        return avg_x, avg_y

    def detect_pinch(self, landmarks):
        """Detect pinch gesture"""
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_FINGER_TIP]

        distance = np.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2)
        return distance < self.pinch_threshold

    def detect_gesture(self, landmarks):
        """Recognize hand gesture"""
        # Check if fingers are extended
        fingers = []

        # Thumb (check x distance from pinky base)
        thumb_tip = landmarks[THUMB_TIP]
        thumb_ip = landmarks[THUMB_IP]
        fingers.append(abs(thumb_tip.x - thumb_ip.x) > 0.03)

        # Other 4 fingers (tip above pip = extended)
        tips = [INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]
        pips = [INDEX_FINGER_PIP, MIDDLE_FINGER_PIP, RING_FINGER_PIP, PINKY_PIP]

        for tip, pip in zip(tips, pips):
            fingers.append(landmarks[tip].y < landmarks[pip].y)

        # Classify gesture
        if all(fingers):
            return "Open Hand ✋"
        elif not any(fingers):
            return "Fist ✊"
        elif fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
            return "Peace ✌️"
        elif self.detect_pinch(landmarks):
            return "Pinch 👌"
        else:
            return "Tracking 👆"

    def draw_landmarks(self, frame, landmarks):
        """Draw hand skeleton on frame"""
        h, w = frame.shape[:2]

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)

        # Draw joints
        for i, lm in enumerate(landmarks):
            x, y = int(lm.x * w), int(lm.y * h)

            # Different colors for fingertips
            if i in [THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]:
                color = (0, 255, 0)
                radius = 6
            elif i == WRIST:
                color = (255, 0, 0)
                radius = 8
            else:
                color = (200, 200, 200)
                radius = 4

            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 1)

    def process_new_api(self, frame):
        """Process using new Tasks API"""
        from mediapipe import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=Image.Format.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        mouse_x = mouse_y = None
        is_pinching = False
        gesture = "None"

        if results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                landmarks = hand_lms

                # Get index finger tip for cursor
                index_tip = landmarks[INDEX_FINGER_TIP]

                # Map to screen (flip X for mirror effect)
                x = (1 - index_tip.x) * self.sensitivity
                y = index_tip.y * self.sensitivity

                # Clamp
                x = max(0, min(1, x))
                y = max(0, min(1, y))

                # Smooth
                x, y = self.smooth_position(x, y)

                # Convert to screen coords
                mouse_x = int(x * self.screen_width)
                mouse_y = int(y * self.screen_height)

                # Detect pinch
                is_pinching = self.detect_pinch(landmarks)
                gesture = self.detect_gesture(landmarks)

                # Draw
                self.draw_landmarks(frame, landmarks)

                # Highlight cursor finger
                h, w = frame.shape[:2]
                tip_x = int(index_tip.x * w)
                tip_y = int(index_tip.y * h)
                cv2.circle(frame, (tip_x, tip_y), 15, (0, 0, 255), 3)

                # Draw pinch line
                if is_pinching:
                    thumb_x = int(landmarks[THUMB_TIP].x * w)
                    thumb_y = int(landmarks[THUMB_TIP].y * h)
                    cv2.line(frame, (tip_x, tip_y), (thumb_x, thumb_y), (0, 255, 255), 4)

                break  # Only first hand

        return frame, mouse_x, mouse_y, is_pinching, gesture

    def process_old_api(self, frame):
        """Process using old Solutions API"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        mouse_x = mouse_y = None
        is_pinching = False
        gesture = "None"

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                landmarks = hand_lms.landmark

                index_tip = landmarks[INDEX_FINGER_TIP]

                x = (1 - index_tip.x) * self.sensitivity
                y = index_tip.y * self.sensitivity
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                x, y = self.smooth_position(x, y)

                mouse_x = int(x * self.screen_width)
                mouse_y = int(y * self.screen_height)

                is_pinching = self.detect_pinch(landmarks)
                gesture = self.detect_gesture(landmarks)

                # Draw using MediaPipe drawing utils
                self.mp_draw.draw_landmarks(
                    frame, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(255, 140, 0), thickness=2)
                )

                h, w = frame.shape[:2]
                tip_x = int(index_tip.x * w)
                tip_y = int(index_tip.y * h)
                cv2.circle(frame, (tip_x, tip_y), 15, (0, 0, 255), 3)

                if is_pinching:
                    thumb_x = int(landmarks[THUMB_TIP].x * w)
                    thumb_y = int(landmarks[THUMB_TIP].y * h)
                    cv2.line(frame, (tip_x, tip_y), (thumb_x, thumb_y), (0, 255, 255), 4)

                break

        return frame, mouse_x, mouse_y, is_pinching, gesture

    def process_frame(self, frame):
        """Main frame processor"""
        frame = cv2.flip(frame, 1)

        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]

        if self.using_new_api:
            return self.process_new_api(frame)
        else:
            return self.process_old_api(frame)

    def draw_ui(self, frame, mouse_x, mouse_y, is_pinching, gesture, fps):
        """Draw user interface overlay"""
        h, w = frame.shape[:2]

        # Background panel
        panel_w, panel_h = 320, 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 255, 0), 2)

        # Status
        tracking = mouse_x is not None
        status = "TRACKING ✓" if tracking else "NO HAND ✗"
        color = (0, 255, 0) if tracking else (0, 0, 255)

        cv2.putText(frame, f"Status: {status}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Gesture: {gesture}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"Sensitivity: {self.sensitivity:.1f}x", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Click indicator
        if is_pinching:
            cv2.circle(frame, (w - 60, 60), 25, (0, 255, 255), -1)
            cv2.circle(frame, (w - 60, 60), 25, (255, 255, 255), 3)
            cv2.putText(frame, "CLICK!", (w - 110, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Cursor position indicator on frame
        if tracking:
            # Show where cursor is on screen (scaled to frame)
            cursor_x = int((mouse_x / self.screen_width) * w)
            cursor_y = int((mouse_y / self.screen_height) * h)
            cv2.circle(frame, (cursor_x, cursor_y), 8, (255, 0, 255), 2)
            cv2.circle(frame, (cursor_x, cursor_y), 3, (255, 0, 255), -1)

        # Instructions at bottom
        cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Q=Quit  S=Sensitivity  R=Reset  |  Pinch=Click  |  Move hand=Cursor",
                   (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        """Main application loop"""
        print("=" * 65)
        print("🖐️  HAND MOUSE CONTROLLER")
        print("=" * 65)
        print("Controls:")
        print("  👆 Move your INDEX FINGER to control the cursor")
        print("  👌 PINCH thumb + index finger to CLICK")
        print("  ⌨️  Q = Quit  |  S = Toggle Sensitivity  |  R = Reset")
        print("=" * 65)

        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ ERROR: Cannot open camera!")
            print("Check if your webcam is connected and not in use by another app.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # FPS tracking
        fps_history = deque(maxlen=30)
        prev_time = time.time()

        print("\n🚀 Starting in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        print("\n✅ ACTIVE! Show your hand to the camera.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # FPS
                now = time.time()
                fps = 1.0 / (now - prev_time)
                fps_history.append(fps)
                avg_fps = sum(fps_history) / len(fps_history)
                prev_time = now

                # Process hand
                frame, mouse_x, mouse_y, is_pinching, gesture = self.process_frame(frame)

                # Control mouse
                if mouse_x is not None and mouse_y is not None:
                    pyautogui.moveTo(mouse_x, mouse_y, duration=0.005)

                    # Handle click with cooldown
                    if is_pinching and not self.is_pinching:
                        if now - self.last_click_time > self.click_cooldown:
                            pyautogui.click()
                            self.last_click_time = now
                            self.is_pinching = True
                    elif not is_pinching:
                        self.is_pinching = False

                # Draw UI
                frame = self.draw_ui(frame, mouse_x, mouse_y, is_pinching, gesture, avg_fps)

                # Show
                cv2.imshow("🖐️ Hand Mouse Controller", frame)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    self.sensitivity = 1.2 if self.sensitivity > 1.5 else 2.0
                    print(f"Sensitivity: {self.sensitivity}x")
                elif key == ord('r') or key == ord('R'):
                    self.position_buffer.clear()
                    print("Smoothing reset")

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'detector') and self.using_new_api:
                self.detector.close()
            elif hasattr(self, 'hands') and not self.using_new_api:
                self.hands.close()
            print("\n✅ Hand Mouse Controller closed.")


if __name__ == "__main__":
    # Safety: move mouse to corner to emergency stop
    pyautogui.FAILSAFE = True

    controller = HandMouseController()
    controller.run()
