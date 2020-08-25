from game import Tetromino, pieces, templatenum, blank, black
import copy
import pygame
from pygame.locals import *
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import count
from collections import deque
from collections import namedtuple
import os, math, random


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
        self.calc_reward = 0.0 
    
    def step(self, action, needdraw=True):
        is_terminal = False
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

        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.tetromino.addtoboard(self.board,self.fallpiece)
            self.score += self.tetromino.removecompleteline(self.board)            
            self.level, self.fallfreq = self.tetromino.calculate(self.score)   

            self.fallpiece = None
        else:
            self.fallpiece['y'] +=1

        if needdraw:
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
                is_terminal = True       
                self.reset()     
                return is_terminal
        return is_terminal

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
        # 需要加上当前下落方块的值
        if self.fallpiece == None:
            piece = self.fallpiece
            shapedraw = pieces[piece['shape']][piece['rotation']]
            for x in range(templatenum):
                for y in range(templatenum):
                    if shapedraw[y][x]!=blank:
                        board[x][y]=1
        return torch.tensor(board, dtype=torch.float).view(1,-1)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

BATCH_SIZE = 512
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 10000.
TARGET_UPDATE = 10
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

n_actions = 4 
buffer = deque(maxlen=10000)
modle_file = 'data/save/05_02_checkpoint.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps_done = 0
# 200 是面板数据 10*20 ，512 是隐藏层大小
net = Net(200, 512, n_actions)
optimizer = optim.Adam(net.parameters(), lr=1e-5)

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            # t.max(1)将返回每行的最大列值。 
            # 最大结果的第二列是找到最大元素的索引，因此我们选择具有较大预期奖励的行动。
            return net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(buffer) < BATCH_SIZE:
        return
    transitions = random.sample(buffer, BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                      batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train(agent):
    global GAMMA, steps_done
    num_episodes = 5000000
    avg_step = 10.
    need_draw = False
    step_episode_update = 0.
    
    # 加载模型
    if os.path.exists(modle_file):
        checkpoint = torch.load(modle_file)
        net_sd = checkpoint['net']
        steps_done = checkpoint['steps_done']
        avg_step = checkpoint['avg_step']
        net.load_state_dict(net_sd)
            
    for i_episode in range(num_episodes):
        avg_loss = 0.
        state = agent.getBoard().to(device)
        for t in count():
            if need_draw:
                for event in pygame.event.get():  # 需要事件循环，否则白屏
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()  

            action = select_action(state)
            action_value = action.item()
            is_terminal = agent.step(action_value, need_draw)

            if is_terminal:
                _reward = -1.0
                next_state = None
            else:
                _reward = math.exp(-1. * t / avg_step)    
                next_state = agent.getBoard().to(device)
            
            reward = torch.tensor([_reward], device=device)
            buffer.append(Transition(state, action, next_state, reward))
            state = next_state

            loss = optimize_model()
            if loss!=None:       
                avg_loss += loss.item() 

            if is_terminal or t>=10000:
                break
        
        step_episode_update += t
        avg_loss = avg_loss / t
        if avg_loss>1: 
            GAMMA = GAMMA * 0.999
        elif avg_loss<0.01:
            GAMMA = min(GAMMA * 1.001, 0.999)  

        if i_episode % TARGET_UPDATE == 0:
            avg_step = avg_step*0.99 + step_episode_update/TARGET_UPDATE*0.01 
            print(i_episode, steps_done, "%.2f/%.2f"%(step_episode_update/TARGET_UPDATE, avg_step), \
                "loss:", avg_loss, \
                "action_random: %.2f"%(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)), \
                "GAMMA:", GAMMA )
            step_episode_update = 0.
            torch.save({'net': net.state_dict(),
                        'steps_done': steps_done,
                        'avg_step': avg_step,
                        }, modle_file)

if __name__ == "__main__":
    tetromino = Tetromino()
    agent = Agent(tetromino)
    train(agent)