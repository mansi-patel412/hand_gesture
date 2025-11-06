import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]
cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def detect_gesture(fingers, lmList, all_hands):
    # Single-hand gestures
    if fingers == [0,0,0,0,0]:
        return "Fist ✊"
    elif fingers == [1,1,1,1,1]:
        return "Open Hand ✋"
    elif fingers == [1,0,0,0,0]:
        return "Thumbs Up 👍"
    elif fingers == [0,0,0,0,1]:
        return "Thumbs Down 👎"
    elif fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
        return "Peace ✌️"
    elif fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "OK 👌"
    elif fingers[1] and not fingers[0] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "Finger Point 👉"
    elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "Finger Gun 🔫"

    # Rock-Paper-Scissors
    if fingers == [0,1,1,0,0]:
        return "Scissors ✌️"
    elif fingers == [0,1,1,1,1]:
        return "Paper ✋"
    elif fingers == [0,0,0,0,0]:
        return "Rock ✊"

    # Two-hand gestures
    if len(all_hands) == 2:
        h1, h2 = all_hands[0], all_hands[1]
        # Cheers gesture
        if distance(h1[8], h2[8]) < 80:
            return "Cheers 🍻"
        # Heart gesture: thumbs + index fingers close together
        heart_dist = distance(h1[4], h2[4]) + distance(h1[8], h2[8])
        if heart_dist < 160:
            return "Heart ❤️"

    return "Unknown ❓"

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    all_hands = []

    if result.multi_hand_landmarks:
        # Store landmarks for two-hand gestures
        for handLms in result.multi_hand_landmarks:
            h, w, c = img.shape
            lmList = [(int(lm.x*w), int(lm.y*h)) for lm in handLms.landmark]
            all_hands.append(lmList)

        for i, lmList in enumerate(all_hands):
            hand_label = result.multi_handedness[i].classification[0].label
            fingers = []

            # Thumb
            if hand_label == "Right":
                fingers.append(1 if lmList[tipIds[0]][0] > lmList[tipIds[0]-1][0] else 0)
            else:
                fingers.append(1 if lmList[tipIds[0]][0] < lmList[tipIds[0]-1][0] else 0)

            # Other fingers
            for id in range(1,5):
                fingers.append(1 if lmList[tipIds[id]][1] < lmList[tipIds[id]-2][1] else 0)

            gesture = detect_gesture(fingers, lmList, all_hands)
            mp_draw.draw_landmarks(img, result.multi_hand_landmarks[i], mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, f'{hand_label}: {gesture}', (10, 50 + i*50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Hand Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

