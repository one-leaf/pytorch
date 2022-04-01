# https://google.github.io/mediapipe/solutions/hands.html

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

DrawingSpec_point = mp_drawing.DrawingSpec((0,255,0), 2, 2)
DrawingSpec_line = mp_drawing.DrawingSpec((0,0,255), 2, 2)

mp_hands = mp.solutions.hands

hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while cap.isOpened():
    ret, image = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands_model.process(image1)
    # print('Handedness:', results.multi_handedness)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print('hand_landmarks:', hand_landmarks)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands_model.close()
cv2.destroyAllWindows()
cap.release()