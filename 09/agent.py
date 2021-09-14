from numpy.core.shape_base import stack
from game import Tetromino, TetrominoEnv, pieces, templatenum, blank 
from game import calcReward, rowTransitions, colTransitions, emptyHoles, wellNums, landingHeight 
import pygame
from pygame.locals import *
from itertools import count
import numpy as np
import copy
import random

KEY_NONE, KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN = 0, 1, 2, 3, 4
ACTIONS = [KEY_NONE, KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
ACTIONS_NAME = ["N","O","L","R","D"]
class Agent(object):
    def __init__(self):
        self.width = 10
        self.height = 20
        self.actions_num = len(ACTIONS)    
        # self.lock = random.choice([0,1])  
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
        # 当前player
        self.curr_player = 0         
        # 每个方块的高度
        self.pieces_height = []     
        # 下降的状态
        self.fallpiece_status = [self.get_fallpiece_board()]
        # 忽略的步骤
        self.ig_action = None
        # 下一个可用步骤
        self.availables=self.get_availables()
        # 最大游戏高度
        self.limit_max_height = -1
        # 游戏动作
        # self.actions=[]

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
        # if self.curr_player == self.lock:
        #     return [KEY_DOWN, KEY_NONE]
        # else:
        #     acts.remove(KEY_DOWN)

        if not self.tetromino.validposition(self.board,self.fallpiece,ax = -1):
            acts.remove(KEY_LEFT)
        if not self.tetromino.validposition(self.board,self.fallpiece,ax = 1):
            acts.remove(KEY_RIGHT)   
        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            acts.remove(KEY_DOWN)

        # 只允许前面旋转
        # if self.piecesteps>len(pieces[self.fallpiece['shape']]):
        #     acts.remove(KEY_ROTATION)
        # else:
        if self.fallpiece['shape']=="o":
            acts.remove(KEY_ROTATION)
        else:
            r = self.fallpiece['rotation']
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.tetromino.validposition(self.board,self.fallpiece):
                acts.remove(KEY_ROTATION)
            self.fallpiece['rotation'] = r

        random.shuffle(acts)
        if self.ig_action!=None and len(acts)>=2:
            if self.ig_action in acts:
                acts.remove(self.ig_action)

        return acts         

    def game_end(self):
        # return self.terminal, self.lock
        return self.terminal, self.curr_player
        # lastplayer = (self.curr_player+1) % 2
        # return self.terminal, lastplayer

    def step(self, action, env=None):
        # 状态 0 下落过程中 1 更换方块 2 结束一局
        
        self.reward = 0
        self.steps += 1
        self.piecesteps += 1
        self.level, self.fallfreq = self.tetromino.calculate(self.score)
        self.curr_player = self.steps%2
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

        if self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.fallpiece['y'] +=1

        fallpiece_y = self.fallpiece['y']

        self.fallpiece_status.append(self.get_fallpiece_board())

        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.tetromino.addtoboard(self.board,self.fallpiece)
            self.reward = self.tetromino.removecompleteline(self.board) 
            
            # if self.reward >0:
            #     self.terminal = True 
            #     self.state = 2       
            #     return self.state, self.reward
            
            self.score += self.reward          
            # self.level, self.fallfreq = self.tetromino.calculate(self.score)   
            # self.fallpiece_height = landingHeight(self.fallpiece)
            r = 0.5 if self.reward>0 else 0
            self.pieces_height.append(20 - fallpiece_y - r)
            self.fallpiece = None

        if  env:
            env.checkforquit()
            env.render(self.board, self.score, self.level, self.fallpiece, self.nextpiece)

        if self.fallpiece == None:
            self.fallpiece = self.nextpiece
            self.nextpiece = self.tetromino.getnewpiece()
            self.piecesteps = 0
            self.piececount +=1 

            if (not self.tetromino.validposition(self.board,self.fallpiece)) or \
                (self.limit_max_height>0 and self.getMaxHeight()>self.limit_max_height):  
                
                self.terminal = True 
                self.state = 2
                self.reward = -1      
                self.availables = []
                # print(">>>>>", len(self.pieces_height), self.pieces_height, self.actions[-10:])
                return self.state, self.reward 
            else: 
                self.state = 1

            self.fallpiece_status=[self.get_fallpiece_board()]          
        else:
            self.state = 0
        
        # 早期训练中，如果得分就表示游戏结束
        # if self.reward!=0: 
        #     self.terminal=True

        self.availables = self.get_availables()

        return self.state, self.reward

    def get_key(self):
        info = self.getBoard() + self.fallpiece_status[-1]
        if self.curr_player==1:
            info = info + np.ones((self.height, self.width))
        return hash(info.data.tobytes())        
        # key = [0 for v in range(self.height*self.width)]
        # for x in range(self.height):
        #     for y in range(self.width):
        #         if info[x][y]==0:
        #             key[x*self.width+y]='0'
        #         else:
        #             key[x*self.width+y]='1'
        # key3 = int("".join(key),2)
        # return hash(key3)

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
    # 背景 + 前6步走法 = 9
    # 返回 [9, height, width]
    def current_state(self):
        state = np.zeros((9, self.height, self.width))
        state[0] = self.getBoard()
        for j, fallpices in enumerate(self.fallpiece_status[-8:]):
            state[j+1]=fallpices
        return state          


    # 使用 mcts 训练，重用搜索树，并保存数据
    def start_self_play(self, player, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        
        # if self.limit_max_height > 0:
        #     limit_max_height = self.limit_max_height
        # else:
        #     limit_max_height = random.randint(5,12)
        #     self.limit_max_height = limit_max_height

        # print("limit_max_height:", limit_max_height)

        game_num = 2
        self.limit_max_height = random.randint(1,20)
        # limit_max_height = self.limit_max_height
        game_states, game_mcts_probs, game_current_players = [],[],[] 
        game_piececount, game_score, game_winer = [],[],[]
        for _ in range(game_num):
            # self.lock = (self.lock + 1)%2 
            # self.availables = self.get_availables()
            print("limit_max_height", self.limit_max_height)

            _states, _mcts_probs, _current_players=[],[],[]
            game = copy.deepcopy(self)
            # game.limit_max_height = 5
            _piecestep=0
            for i in count():
                action, move_probs = player.get_action(game, temp=temp, return_prob=1) 

                _states.append(game.current_state())
                _mcts_probs.append(move_probs)
                #_current_players.append(game.curr_player)

                game.step(action)
                _piecestep+=1

                if game.state!=0:
                    # game.limit_max_height = max(game.pieces_height)+3
                    # if game.limit_max_height>limit_max_height: game.limit_max_height=limit_max_height
                    print('reward:',game.reward, 'len:', len(game.pieces_height), "max_height:", game.getMaxHeight(), "next:", game.fallpiece['shape'], game.pieces_height)
                    temp_winer = []
                    winer = 1 if game.reward>0 else -1
                    for _ in range(_piecestep):
                        temp_winer.insert(0,winer)
                        winer = -1 * winer
                    _current_players.extend(temp_winer)
                    _piecestep = 0

                if game.terminal:
                    break
            
            game_states.append(_states)
            game_mcts_probs.append(_mcts_probs)
            game_current_players.append(_current_players)

            game_piececount.append(game.piececount)
            game_score.append(game.score)
            _, winer = game.game_end()
            game_winer.append(winer)
            game.print()

        # max_score = max(game_score)
        # game_player_0 = [-1 for _ in range(game_num)] 
        # game_player_1 = [-1 for _ in range(game_num)] 

        # min_game = -1
        # max_game = -1

        # min_piececount = min(game_piececount)
        # max_piececount = max(game_piececount)

        # if game_piececount.count(min_piececount)==1:
        #     min_game = game_piececount.index(min_piececount)

        # if game_piececount.count(max_piececount)==1:
        #     max_game = game_piececount.index(max_piececount)

        # for j in range(game_num):
        #     game_player_0[j] = 1 if game_winer[j]==0 else -1
        #     game_player_1[j] = 1 if game_winer[j]==1 else -1

        # sort_index = sorted(range(len(game_piececount)), key=lambda k: game_piececount[k]+game_score[k])
        
        # split_index = -(game_num//2)
        # max_index = sort_index[split_index:]
        # min_index = sort_index[:split_index]

        # print("game_piececount",game_piececount,"game_score",game_score,"max",max_game,"min",min_game)
        # print("sort_index", sort_index, "max_index", max_index, "min_index", min_index)
        # print("game_player_0",game_player_0,"game_player_1",game_player_1)


        states, mcts_probs, winers= [], [], []
        for j in range(game_num):
            for o in game_states[j]: states.append(o)
            for o in game_mcts_probs[j]: mcts_probs.append(o)
            for o in game_current_players[j]: winers.append(o)
            # if j in min_index:
            #     for p in game_current_players[j]:
            #         winers.append(-1)
            # elif j in max_index:
            #     for p in game_current_players[j]:
            #         winers.append(1)
            # else:
            # for p in game_current_players[j]:
            #     if p==0:
            #         winers.append(game_player_0[j])
            #     else:
            #         winers.append(game_player_1[j])

        winners_z = np.array(winers)

        assert len(states)==len(mcts_probs)
        assert len(states)==len(winners_z)

        _len = len(winers)
        _win = (_len+sum(winers))/2
        _loss = _len - _win 
        print("add %s to dataset，winner: %s, loss: %s"%(_len,_win,_loss))
        reward, piececount, agentcount = 0, 0, 0
        reward = sum(game_score)  
        piececount = sum(game_piececount)
        agentcount = game_num
    
        return reward, piececount, agentcount, zip(states, mcts_probs, winners_z)
                
