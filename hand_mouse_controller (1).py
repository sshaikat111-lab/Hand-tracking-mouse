import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ================= SETTINGS =================

SMOOTHING = 15
CLICK_DISTANCE = 20
FRAME_REDUCTION = 80
DEAD_ZONE = 8

pyautogui.FAILSAFE = True

screen_w, screen_h = pyautogui.size()

# ================= CAMERA =================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ================= MEDIAPIPE =================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)

mp_draw = mp.solutions.drawing_utils

# ================= VARIABLES =================

prev_x, prev_y = 0, 0
last_click_time = 0
p_time = 0

# ================= FUNCTIONS =================

def distance(p1, p2):
    return math.hypot(
        p2[0] - p1[0],
        p2[1] - p1[1]
    )

# ================= MAIN LOOP =================

while True:

    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    cv2.rectangle(
        frame,
        (FRAME_REDUCTION, FRAME_REDUCTION),
        (w - FRAME_REDUCTION, h - FRAME_REDUCTION),
        (255, 0, 255),
        2
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        lm = []

        for landmark in hand_landmarks.landmark:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            lm.append((x, y))

        thumb_tip = lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]

        # ==========================================
        # STABLE CURSOR CONTROL USING HAND CENTER
        # ==========================================

        hand_x = (
            lm[5][0] +
            lm[9][0] +
            lm[13][0]
        ) / 3

        hand_y = (
            lm[5][1] +
            lm[9][1] +
            lm[13][1]
        ) / 3

        mouse_x = (
            (hand_x - FRAME_REDUCTION)
            * screen_w
            / (w - 2 * FRAME_REDUCTION)
        )

        mouse_y = (
            (hand_y - FRAME_REDUCTION)
            * screen_h
            / (h - 2 * FRAME_REDUCTION)
        )

        mouse_x = max(
            0,
            min(screen_w, mouse_x)
        )

        mouse_y = max(
            0,
            min(screen_h, mouse_y)
        )

        # Ignore tiny shakes

        if abs(mouse_x - prev_x) < DEAD_ZONE:
            mouse_x = prev_x

        if abs(mouse_y - prev_y) < DEAD_ZONE:
            mouse_y = prev_y

        # Smooth movement

        curr_x = prev_x + (
            mouse_x - prev_x
        ) / SMOOTHING

        curr_y = prev_y + (
            mouse_y - prev_y
        ) / SMOOTHING

        pyautogui.moveTo(curr_x, curr_y)

        prev_x = curr_x
        prev_y = curr_y

        # ==========================================
        # LEFT CLICK
        # ==========================================

        if distance(
            thumb_tip,
            index_tip
        ) < CLICK_DISTANCE:

            if (
                time.time()
                - last_click_time
            ) > 0.5:

                pyautogui.click()

                last_click_time = time.time()

        # ==========================================
        # RIGHT CLICK
        # ==========================================

        if distance(
            thumb_tip,
            middle_tip
        ) < CLICK_DISTANCE:

            if (
                time.time()
                - last_click_time
            ) > 0.5:

                pyautogui.rightClick()

                last_click_time = time.time()

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    # ================= FPS =================

    c_time = time.time()

    fps = (
        1 / (c_time - p_time)
        if p_time != 0
        else 0
    )

    p_time = c_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Hand Mouse Pro",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================

cap.release()
cv2.destroyAllWindows()
