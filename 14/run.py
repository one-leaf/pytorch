# https://google.github.io/mediapipe/solutions/hands.html

import cv2
import mediapipe as mp
import os
from model import Model, RNN, getLabelName, getWin, getWinAction
import torch
import random
from collections import deque
import time

mp_drawing = mp.solutions.drawing_utils

DrawingSpec_point = mp_drawing.DrawingSpec((0,255,0), 2, 2)
DrawingSpec_line = mp_drawing.DrawingSpec((0,0,255), 2, 2)

mp_hands = mp.solutions.hands

hands_model = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

curr_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(curr_dir, 'model')
model_file = os.path.join(model_dir, 'model.pth')

model = Model()
model.load_state_dict(torch.load(model_file))

rnn_model_file = os.path.join(model_dir, 'rnn_model.pth')
rnn = RNN()
if os.path.exists(rnn_model_file):
    rnn.load_state_dict(torch.load(rnn_model_file))
h_state = None
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
rnn_next_pred = None
train_queue = deque(maxlen=64)

# 猜拳的结果 人赢的局数，机器赢的局数，平局的局数
win_counts = [0, 0, 0]

input("Are you ready? Press Enter to start.")

while True: 

    while cap.isOpened():
        time.sleep(1)    

        ret, image = cap.read()
        ret, image = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands_model.process(image1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

                # 预测出拳    
                info = []
                for i in range(0, len(mp.solutions.hands.HandLandmark)):
                    info.append(hand_landmarks.landmark[i].x)
                    info.append(hand_landmarks.landmark[i].y)
                    info.append(hand_landmarks.landmark[i].z)
                if len(info)!=63: continue
                info = torch.tensor(info).view(1, -1)
                pred = model(info)
                pred_idx = pred.argmax(dim=1).item()
                pred_name = getLabelName(pred_idx)
                train_queue.append(pred_idx)

                # 获取下次的出拳
                if rnn_next_pred==None:
                    pc_action = random.randint(0, 2)  
                else:
                    pc_action = getWinAction(rnn_next_pred_idx)
                pc_action_name = getLabelName(pc_action)                

                # 计算猜拳的结果
                game_win = getWin(pred_idx, pc_action)
                if game_win==1:
                    win_counts[0]+=1
                elif game_win==-1:
                    win_counts[1]+=1
                else:
                    win_counts[2]+=1
                print("you:",pred_name,"pc:",pc_action_name,"game_win",game_win,"you win:",win_counts[0],"pc win:",win_counts[1],"draw:",win_counts[2])

                # 反向传播更新网络参数
                if rnn_next_pred!=None:
                    
                    if len(train_queue)>=2:
                        time_len = len(train_queue)-1 
                        x = torch.zeros((1, time_len ,3))
                        for i in range(0, time_len):
                            x[0][i][train_queue[i]] = 1

                        y = torch.zeros((time_len), dtype=torch.long)
                        for i in range(0, time_len):
                            y[i] = train_queue[i+1]
                        train_pre_y, h_state = rnn(x, h_state)
                        optimizer.zero_grad()
                        train_pre_y = train_pre_y.view((time_len,3))
                        loss =criterion(train_pre_y, y)
                        loss.backward()
                        optimizer.step()
                        print(loss.item())
                        torch.save(rnn.state_dict(), rnn_model_file)

                # 获取下次的出拳
                time_len = len(train_queue)
                x = torch.zeros((1, time_len ,3))
                for i in range(0, time_len):
                    x[0][i][train_queue[i]]=1
                with torch.no_grad():
                    rnn_next_pred, h_state = rnn(x, h_state)
                rnn_next_pred = rnn_next_pred.view((-1,3))[-1]
                rnn_next_pred_idx = rnn_next_pred.argmax(dim=0).item()

                cv2.imshow('image', image)
                cv2.waitKey(0) 

            break
        
hands_model.close()
cap.release()