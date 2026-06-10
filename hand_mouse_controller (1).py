import cv2
import pyautogui
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8, maxHands=1)

screen_w, screen_h = pyautogui.size()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        index_x, index_y = lmList[8][0], lmList[8][1]

        cam_h, cam_w, _ = img.shape

        screen_x = int(index_x * screen_w / cam_w)
        screen_y = int(index_y * screen_h / cam_h)

        pyautogui.moveTo(screen_x, screen_y)

        fingers = detector.fingersUp(hand)

        # Index + Middle finger together = Click
        if fingers[1] == 1 and fingers[2] == 1:
            pyautogui.click()

    cv2.imshow("Hand Mouse Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()import cv2
import pyautogui
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8, maxHands=1)

screen_w, screen_h = pyautogui.size()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        index_x, index_y = lmList[8][0], lmList[8][1]

        cam_h, cam_w, _ = img.shape

        screen_x = int(index_x * screen_w / cam_w)
        screen_y = int(index_y * screen_h / cam_h)

        pyautogui.moveTo(screen_x, screen_y)

        fingers = detector.fingersUp(hand)

        # Index + Middle finger together = Click
        if fingers[1] == 1 and fingers[2] == 1:
            pyautogui.click()

    cv2.imshow("Hand Mouse Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
