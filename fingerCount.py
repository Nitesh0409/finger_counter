import time
import cv2
import mediapipe as mp
import math

# distance calculation
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# webcam
cap = cv2.VideoCapture(0)

# mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

currTime = 0
prevTime = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera not connected.")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        total_fingers = 0
        for hand_index, (hand_landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
            lm_list = []
            for idx, hl in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(hl.x * w), int(hl.y * h)
                lm_list.append((idx, cx, cy))

            finger_up = []

            finger_tips = [4, 8, 12, 16, 20]

            if lm_list:
                for tip in finger_tips:
                    ax, ay = lm_list[tip][1], lm_list[tip][2]
                    bx, by = lm_list[tip - 1][1], lm_list[tip - 1][2]
                    cx, cy = lm_list[tip - 2][1], lm_list[tip - 2][2]
                    dx, dy = lm_list[tip - 3][1], lm_list[tip - 3][2]

                    ab = distance(ax, ay, bx, by)
                    bc = distance(bx, by, cx, cy)
                    cd = distance(cx, cy, dx, dy)
                    ad = distance(ax, ay, dx, dy)

                    if abs((ab + bc + cd) - ad) < 3:
                        finger_up.append(tip)
                        cv2.circle(frame, (ax, ay), 20, (50, 230, 230), -1)

            label = handedness.classification[0].label
            x0, y0 = lm_list[0][1], lm_list[0][2]
            cv2.putText(frame, f'{label} Hand: {len(finger_up)}', (x0 - 30, y0 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 50, 50), 2)

            total_fingers += len(finger_up)
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

        cv2.putText(frame, f'Total Fingers: {total_fingers}', (300, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 150, 255), 3)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(frame, f'{int(fps)} FPS', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Session ended by user.")
        break

cap.release()
cv2.destroyAllWindows()
