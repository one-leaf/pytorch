# https://google.github.io/mediapipe/solutions/hands.html

import cv2
import mediapipe as mp
import os,uuid,json

mp_drawing = mp.solutions.drawing_utils

DrawingSpec_point = mp_drawing.DrawingSpec((0,255,0), 2, 2)
DrawingSpec_line = mp_drawing.DrawingSpec((0,0,255), 2, 2)

mp_hands = mp.solutions.hands

hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


curr_dir = os.path.dirname(os.path.abspath(__file__))
# 石头 剪刀 布 的训练数据目录
data_dir = os.path.join(curr_dir, 'data')


curr_data_type = '石头'  #  石头,  剪刀,  布

no = 0
while cap.isOpened():
    ret, image = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands_model.process(image1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            info = []
            for i in range(0, len(mp.solutions.hands.HandLandmark)):
                info.append(hand_landmarks.landmark[i].x)
                info.append(hand_landmarks.landmark[i].y)
                info.append(hand_landmarks.landmark[i].z)
            print(len(info), info)
            if len(info)!=63: continue
            filename = os.path.join(data_dir, curr_data_type, '%s_%s.json' % (curr_data_type, uuid.uuid1()))
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            json.dump(info, open(filename, 'w'))
            no += 1
            print("save %s" % filename)

hands_model.close()
cv2.destroyAllWindows()
cap.release()