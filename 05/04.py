from os import stat
from time import time
from numpy.lib.stride_tricks import broadcast_arrays
from game import Tetromino, pieces, templatenum, blank, black
import copy
import pygame
from pygame.locals import *
import sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import count
from collections import deque
from collections import namedtuple
import os, math, random

import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

KEY_ROTATION  = 0
KEY_LEFT      = 1
KEY_RIGHT     = 2
KEY_DOWN      = 3

class Agent(object):
    def __init__(self, tetromino):
        self.tetromino = tetromino
        self.reset()

    def reset(self):
        self.fallpiece = self.tetromino.getnewpiece()
        self.nextpiece = self.tetromino.getnewpiece()
        self.score = 0
        self.level = 0
        self.board = self.tetromino.getblankboard()
    
    def step(self, action, need_draw=True):
        # 状态 0 下落过程中 1 更换方块 2 结束一局
        state = 0
        reward = 0
        self.level, self.fallfreq = self.tetromino.calculate(self.score)

        if action == KEY_LEFT and self.tetromino.validposition(self.board,self.fallpiece,ax = -1):
            self.fallpiece['x']-=1

        if action == KEY_RIGHT and self.tetromino.validposition(self.board,self.fallpiece,ax = 1):
            self.fallpiece['x']+=1  

        if action == KEY_DOWN and self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.fallpiece['y']+=1  

        if action == KEY_ROTATION:
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.tetromino.validposition(self.board,self.fallpiece):
                self.fallpiece['rotation'] = (self.fallpiece['rotation'] - 1) % len(pieces[self.fallpiece['shape']])
            # if self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            #     self.fallpiece['y']+=1

        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.tetromino.addtoboard(self.board,self.fallpiece)
            reward = self.tetromino.removecompleteline(self.board) 
            self.score += reward          
            self.level, self.fallfreq = self.tetromino.calculate(self.score)   
            self.fallpiece = None
        else:
            self.fallpiece['y'] +=1

        if need_draw:
            self.tetromino.disp.fill(black)
            self.tetromino.drawboard(self.board)
            self.tetromino.drawstatus(self.score, self.level)
            self.tetromino.drawnextpiece(self.nextpiece)
            if self.fallpiece !=None:
                self.tetromino.drawpiece(self.fallpiece)
            pygame.display.update()

        if self.fallpiece == None:
            self.fallpiece = self.nextpiece
            self.nextpiece = self.tetromino.getnewpiece()
            if not self.tetromino.validposition(self.board,self.fallpiece):   
                state = 2       
                return state, reward
            else: 
                state =1
        else:
            state = 0
        return state, reward

    def getBoard(self):
        board=[]
        # 得到当前面板的值
        for line in self.board:
            board.append([])
            for value in line:
                if value==blank:
                    board[-1].append(0)
                else:
                    board[-1].append(1)
        board = torch.tensor(board, dtype=torch.float)
        return board

    def get_fallpiece_board(self):                    
        board=[[0]*20 for _ in range(10)]
        # 需要加上当前下落方块的值
        if self.fallpiece != None:
            piece = self.fallpiece
            shapedraw = pieces[piece['shape']][piece['rotation']]
            for x in range(templatenum):
                for y in range(templatenum):
                    if shapedraw[y][x]!=blank:
                        px, py = x+piece['x'], y+piece['y']
                        if px>=0 and py>=0:
                            board[x+piece['x']][y+piece['y']]=1
        board = torch.tensor(board, dtype=torch.float)
        return board

    # 反向显示
    def get_nextpiece_borad(self):
        board=[[1]*20 for _ in range(10)]
        if self.nextpiece != None:
            piece = self.nextpiece  
            shapedraw = pieces[piece['shape']][piece['rotation']]
            for x in range(templatenum):
                for y in range(templatenum):
                    if shapedraw[y][x]!=blank:
                        board[x][y]=0
        board = torch.tensor(board, dtype=torch.float)
        return board

    # 计算当前的最高点
    def getBoardCurrHeight(self):
        height=len(self.board[0])
        for line in self.board:
            for h, value in enumerate(line): 
                if value!=blank:
                    if h<height:
                        height = h
        return len(self.board[0]) - height

    # 判断是否存在空洞
    def isExitesEmptyHoles(self):
        boardwidth = len(self.board)
        boardheight = len(self.board[0])
        for x in range(boardwidth):
            find_block = False
            for y in range(boardheight):
                if self.board[x][y]!=blank:
                    find_block = True
                elif find_block:
                    return True            
        return False

class Net(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv8 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv9 = nn.Conv2d(32, 1, 1, 1)
        self.fc1 = nn.Linear(10*20, output_size)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x

BATCH_SIZE = 256
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000000.
TARGET_UPDATE = 10
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

n_actions = 4 
buffer = deque(maxlen=100000)
modle_file = 'data/save/05_04_checkpoint.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps_done = 0
# 200 是面板数据 10*20
net = Net(n_actions).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-6)

net_actions_count=torch.tensor([0,0,0,0], device=device, dtype=torch.long)

def select_action(state, norandom=False):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if norandom or random.random() > eps_threshold:
        with torch.no_grad():
            # t.max(1)将返回每行的最大列值。 
            # 最大结果的第二列是找到最大元素的索引，因此我们选择具有较大预期奖励的行动。
            action = net(state).max(1)[1].view(1, 1)
            net_actions_count[action]+=1
            return action
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(buffer) < BATCH_SIZE:
        return
    transitions = random.sample(buffer, BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                      batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train(agent):
    global GAMMA, steps_done
    num_episodes = 100000
    avg_step = 100.
    step_episode_update = 0.
    
    # 加载模型
    if os.path.exists(modle_file):
        checkpoint = torch.load(modle_file, map_location=device)
        net_sd = checkpoint['net']
        steps_done = checkpoint['steps_done']
        avg_step = checkpoint['avg_step']
        net.load_state_dict(net_sd)

    for i_episode in range(num_episodes):
        avg_loss = 0.
        board = agent.getBoard().to(device)
        board_1 = agent.get_fallpiece_board().to(device)
        board_2 = agent.get_nextpiece_borad().to(device)
        state = torch.stack([board,board_1,board_2])
        piece_step = 0  # 方块步数
        for t in count():

            # 前10步都是随机乱走的
            piece_step += 1
            curr_board_height = agent.getBoardCurrHeight()
            if piece_step<10-curr_board_height:
                # action = torch.tensor([[3]], device=device, dtype=torch.long)
                action = torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)
            else:
                action = select_action(state.unsqueeze(0), False)

            action_value = action.item()
            agent_state, _reward = agent.step(action_value, False)

            is_terminal = (agent_state == 2) or (agent_state==1 and agent.isExitesEmptyHoles())

            # 如果是一个新方块落下，设置当前方块的步数为0
            if agent_state==1: 
                piece_step = 0
                               
            # if curr_board_height > 2 + steps_done//1000000:
            #     is_terminal = True

            if is_terminal :
                _reward = -1.
                next_state = None
            else:
                if agent_state==1:
                    if _reward==0:
                        if agent.isExitesEmptyHoles():
                            _reward = -1.
                        else:
                            _reward = -0.5
                    else:
                        _reward += 1.
                else:
                    _reward += 0.5
                board = agent.getBoard().to(device)
                board_1 = agent.get_fallpiece_board().to(device)
                board_2 = agent.get_nextpiece_borad().to(device)
                next_state = torch.stack([board,board_1,board_2])
            
            reward = torch.tensor([_reward], device=device)
            buffer.append(Transition(state, action, next_state, reward))

            # print(t, action, reward)
            # print(t, action, reward,  state)
            # plt.figure()
            # plt.imshow(np.transpose(state,(1,2,0)))
            # if next_state!=None:
            #     plt.figure()            
            #     plt.imshow(np.transpose(next_state,(1,2,0)))
            # plt.show()
            # if  t>3:
            #     raise "ddd"

            loss = optimize_model()
            if loss!=None:       
                avg_loss += loss.item() 

            if is_terminal:
                agent.reset()
                break

            state = next_state

        step_episode_update += t
        avg_step = avg_step*0.999 + t*0.001
        avg_loss = avg_loss / t
        # if avg_loss>1: 
        #     GAMMA = GAMMA * 0.999
        # elif avg_loss<0.1:
        #     GAMMA = min(GAMMA * 1.001, 0.999)  

        if i_episode % TARGET_UPDATE == 0:
            net_actions_count_value = net_actions_count.cpu().numpy()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
                i_episode, steps_done, "%.2f/%.2f"%(step_episode_update/TARGET_UPDATE, avg_step), \
                "loss:", avg_loss, "GAMMA:", GAMMA, \
                "action_counts:",net_actions_count_value/sum(net_actions_count_value) )
            step_episode_update = 0.
            torch.save({'net': net.state_dict(),
                        'steps_done': steps_done,
                        'avg_step': avg_step,
                        }, modle_file)
            if i_episode>0 and i_episode % 1000 == 0:
                torch.save({'net': net.state_dict(),
                        'steps_done': steps_done,
                        'avg_step': avg_step,
                        }, modle_file+"_%s"%steps_done)  

def test(agent):
    # 加载模型
    num_episodes = 10
    checkpoint = torch.load(modle_file, map_location=device)
    net_sd = checkpoint['net']
    net.load_state_dict(net_sd)
    net.eval()
    for i_episode in range(num_episodes):
        for t in count():
            for event in pygame.event.get():  # 需要事件循环，否则白屏
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()    

            board = agent.getBoard().to(device)
            board_1 = agent.get_fallpiece_board().to(device)
            board_2 = agent.get_nextpiece_borad().to(device)
            state = torch.stack([board,board_1,board_2])

            # plt.imshow(np.transpose(state,(1,2,0)))
            # plt.show()

            action = net(state.unsqueeze(0)).max(1)[1].view(1, 1)
            action_value = action.item()
            agent_state, _reward = agent.step(action_value, True)

            if agent_state==2: 
                agent.reset()
                break

            time.sleep(0.01)

if __name__ == "__main__":
    tetromino = Tetromino()
    agent = Agent(tetromino)
    # train(agent)
    if device.type == "cpu":
        test(agent)
    else:
        train(agent)