from numpy.core.shape_base import stack
from torch._C import Value
from game import Tetromino, TetrominoEnv, pieces, templatenum, blank 
from game import calcReward, rowTransitions, colTransitions, emptyHoles, wellNums, landingHeight 
from pygame.locals import *
from itertools import count
import numpy as np
import copy
import random
from collections import deque
import json,os

KEY_NONE, KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN = 0, 1, 2, 3, 4
ACTIONS = [KEY_NONE, KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
ACTIONS_NAME = ["N","O","L","R","D"]
class Agent(object):
    def __init__(self):
        self.width = 10
        self.height = 20
        self.actions_num = len(ACTIONS)    
        self.reset()        

    def reset(self):
        self.tetromino = Tetromino(isRandomNextPiece=False)
        # 下落的方块
        self.fallpiece = self.tetromino.getnewpiece()
        # 下一个待下落的方块
        self.nextpiece = self.tetromino.getnewpiece()
        # 是否结束
        self.terminal = False
        # 得分
        self.score = 0
        # 当前步得分
        self.reward = 0
        # 等级
        self.level = 0
        # 全部步长
        self.steps = 0
        # 每个方块的步长
        self.piecesteps = 0
        # 方块的数量
        self.piececount = 0
        # 面板
        self.board = self.tetromino.getblankboard()
        # 状态： 0 下落过程中 1 更换方块 2 结束一局
        self.state =0
        # 上一个下落方块的截图
        self.prev_fallpiece_boards=None
        # 每个方块的高度
        self.pieces_height = []     
        # 下降的状态
        self.fallpiece_status = deque(maxlen=10)
        for i in range(9):
            self.fallpiece_status.append(np.zeros((self.height, self.width)))
        self.fallpiece_status.append(self.get_fallpiece_board())
        # 下一个可用步骤
        self.availables=self.get_availables()
        # 显示mcts中间过程
        self.show_mcts_process = False

    # 概率的索引位置转action
    def position_to_action(self, position):
        return ACTIONS[position]

    def position_to_action_name(self, position):
        return ACTIONS_NAME[position]

    def positions_to_actions(self, positions):
        return [self.position_to_action(i) for i in positions]

    # action转概率的索引位置
    def action_to_position(self, action):
        return action

    def actions_to_positions(self, actions):
        return [act for act in actions]

    # 获取可用步骤, 保留一个旋转始终有用
    # 将单人游戏变为双人博弈，一个正常下，一个只下走，
    def get_availables(self):
        acts=[KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]

        if not self.tetromino.validposition(self.board,self.fallpiece,ax = -1):
            acts.remove(KEY_LEFT)
        if not self.tetromino.validposition(self.board,self.fallpiece,ax = 1):
            acts.remove(KEY_RIGHT)   
        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            acts.remove(KEY_DOWN)

        if self.fallpiece['shape']=="o":
            acts.remove(KEY_ROTATION)
        else:
            r = self.fallpiece['rotation']
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.tetromino.validposition(self.board,self.fallpiece):
                acts.remove(KEY_ROTATION)
            self.fallpiece['rotation'] = r

        if not KEY_DOWN in acts: acts.append(KEY_NONE)

        random.shuffle(acts)
        
        return acts         


    def game_end(self):
        return self.terminal


    def step(self, action, env=None):
        # 状态 0 下落过程中 1 更换方块 2 结束一局
        
        self.reward = 0
        self.steps += 1
        self.piecesteps += 1
        self.level, self.fallfreq = self.tetromino.calculate(self.score)
        
        # self.actions.append(action)

        if action == KEY_LEFT and self.tetromino.validposition(self.board,self.fallpiece,ax = -1):
            self.fallpiece['x']-=1

        if action == KEY_RIGHT and self.tetromino.validposition(self.board,self.fallpiece,ax = 1):
            self.fallpiece['x']+=1  

        if (action == KEY_DOWN) and self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.fallpiece['y']+=1  

        if action == KEY_ROTATION:
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.tetromino.validposition(self.board,self.fallpiece):
                self.fallpiece['rotation'] = (self.fallpiece['rotation'] - 1) % len(pieces[self.fallpiece['shape']])

        isFalling=True
        if self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.fallpiece['y'] +=1
        else:
            isFalling = False

        fallpiece_y = self.fallpiece['y']

        self.fallpiece_status.append(self.get_fallpiece_board())

        if not isFalling:
            self.tetromino.addtoboard(self.board,self.fallpiece)
            self.reward = self.tetromino.removecompleteline(self.board) 
            
            self.score += self.reward          
            r = 0.5 if self.reward>0 else 0
            self.pieces_height.append(20 - fallpiece_y - r)
            self.fallpiece = None

        if  env:
            env.checkforquit()
            env.render(self.board, self.score, self.level, self.fallpiece, self.nextpiece)

        if not isFalling:
            self.fallpiece = self.nextpiece
            self.nextpiece = self.tetromino.getnewpiece()
            self.piecesteps = 0
            self.piececount +=1 

            if (not self.tetromino.validposition(self.board,self.fallpiece)):                  
                self.terminal = True 
                self.state = 2
                self.availables = [KEY_NONE]
                return self.state, self.reward 
            else: 
                self.state = 1
        else:
            self.state = 0
        
        self.availables = self.get_availables()

        return self.state, self.reward

    def get_key(self):
        info = self.getBoard() + self.fallpiece_status[-1]
        return hash(info.data.tobytes())        


    # 打印
    def print2(self, add_fallpiece=False):
        info = self.getBoard()
        if add_fallpiece:
            info += self.fallpiece_status[-1]
        for y in range(self.height):
            line=str(y%10)+" "
            for x in range(self.width):
                if info[y][x]==0:
                    line=line+"  "
                else:
                    line=line+"* "
            print(line)
        print(" "+" -"*self.width)            
        print("level:", self.level, "score:", self.score, "steps:", self.steps,"piececount:", self.piececount)

    def print(self):
        for y in range(self.height):
            line="| "
            for x in range(self.width):
                if self.board[x][y]==blank:
                    line=line+"  "
                else:
                    line=line+str(self.board[x][y])+" "
            print(line)
        print(" "+" -"*self.width)
        print("level:", self.level, "score:", self.score, "steps:", self.steps,"piececount:", self.piececount)

    # 统计当前最大高度
    def getMaxHeight(self):
        c = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.board[x][y]!=blank:
                    c=y
                    break
            if c!=0:break  
        h = 0 if c == 0 else self.height - c                          
        return h

    # 统计非空的个数
    def getNoEmptyCount(self):
        c = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.board[x][y]!=blank:
                    c+=1
        return c

        
    # 获得当前局面信息
    def getBoard(self):
        board=np.zeros((self.height, self.width))
        # 得到当前面板的值
        for y in range(self.height):
            for x in range(self.width):
                if self.board[x][y]!=blank:
                    board[y][x]=1
        return board

    # 获得下落方块的信息
    def get_fallpiece_board(self):   
        board=np.zeros((self.height, self.width))
        # 需要加上当前下落方块的值
        if self.fallpiece != None:
            piece = self.fallpiece
            shapedraw = pieces[piece['shape']][piece['rotation']]
            for x in range(templatenum):
                for y in range(templatenum):
                    if shapedraw[y][x]!=blank:
                        px, py = x+piece['x'], y+piece['y']
                        if px>=0 and py>=0:
                            board[y+piece['y']][x+piece['x']]=1
        return board

    # 获得待下落方块的信息
    def get_nextpiece_borad(self):
        board=np.zeros((self.height, self.width))
        if self.nextpiece != None:
            piece = self.nextpiece  
            shapedraw = pieces[piece['shape']][piece['rotation']]
            for x in range(templatenum):
                for y in range(templatenum):
                    if shapedraw[y][x]!=blank:
                        board[y][x]=1
        return board

    # 获得当前的全部特征
    # 背景 + 前8步走法 = 9
    # 返回 [9, height, width]
    def current_state(self):
        state = np.zeros((9, self.height, self.width))
        state[0] = self.getBoard()

        for i in range(8):
            state[i+1]=self.fallpiece_status[-1*(i+1)]

        # 前4步是对手的，后4步是自己的
        # for j in range(4): 
        #     idx = -2*j-1  #(-1,-3,-5,-7)
        #     state[j+1]=self.fallpiece_status[idx]
        # for j in range(4):
        #     idx = -2*j-2  #(-2,-4,-6,-8)
        #     state[j+5]=self.fallpiece_status[idx]

        return state          


    # 训练模型
    def start_self_play(self, player, temp=1e-3):
        game_num = 5
        agentcount, agentreward, piececount, agentscore = 0, 0, 0, 0
        game_keys, game_states, game_vals, game_mcts_probs = [], [], [], [] 
        for game_idx in range(game_num):

            _states, _probs, _keys, _masks, _rewards, _qvals = [],[],[],[],[],[]
            game = copy.deepcopy(self)

            if game_idx==0 or game_idx==game_num-1:
                game.show_mcts_process=True
            else:
                game.show_mcts_process=False

            for i in count():

                _keys.append(game.get_key())
                _states.append(game.current_state())
                                
                if game_idx == game_num-1:
                    action, move_probs = player.get_action(game, temp=temp, return_prob=1, need_random=False) 
                else: 
                    action, move_probs = player.get_action(game, temp=temp, return_prob=1, need_random=True) 
               
                _, reward = game.step(action)

                # 这里的奖励是消除的行数
                if reward > 0:
                    _reward = reward * 10
                else:
                    _reward = 0

                # 方块的个数越多越好
                if game.terminal:
                    _reward += game.getNoEmptyCount()               

                _probs.append(move_probs)
                _rewards.append(_reward)
                _masks.append(1-game.terminal)

                if game.terminal:
                    for step in reversed(range(len(_states))):
                        Qval = _rewards[step]
                        _qvals.insert(0, Qval)

                    print(game_idx, 'reward:', game.score, "Qval:", _rewards[-1], 'len:', len(_qvals), "piececount:", game.piececount)
                    agentcount += 1
                    agentscore += game.score
                    agentreward += _reward
                    piececount += game.piececount
                    break

            game_keys.append(_keys)
            game_states.append(_states)
            game_vals.append(_qvals)
            game_mcts_probs.append(_probs)

            game.print()

        avg_agentreward = agentreward / game_num

        for game_idx in range(game_num):
            game_vals[game_idx][-1] -= avg_agentreward
            for i in reversed(range(len(game_keys[game_idx])-1)):
                game_vals[game_idx][i] += game_vals[game_idx][i+1]*0.999  
            print(*game_vals[game_idx][:3], "...", *game_vals[game_idx][-3:])


        keys, states, values, mcts_probs= [], [], [], []
        for j in range(game_num):
            for o in game_keys[j]: keys.append(o)
            for o in game_states[j]: states.append(o)
            for o in game_vals[j]: values.append(o)
            for o in game_mcts_probs[j]: mcts_probs.append(o)

        assert len(states)==len(values)
        assert len(states)==len(keys)
        assert len(states)==len(mcts_probs)
    
        print("add %s to dataset"% len(states) )
    
        return agentcount, agentscore, piececount, keys, zip(states, mcts_probs, values)
                
