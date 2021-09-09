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
        self.tetromino = Tetromino(isRandomNextPiece=False)
        self.width = 10
        self.height = 20
        self.actions_num = len(ACTIONS)    
        self.reset()   

    def reset(self):
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
        # 最大方块数量
        self.limit_piece_count = 0   
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
        if self.curr_player==1: return [KEY_DOWN,]
        # if self.fallpiece['y']>10: return [KEY_NONE,]

        acts=[KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_NONE, KEY_DOWN]

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

        random.shuffle(acts)
        if self.ig_action!=None and len(acts)>=2:
            if self.ig_action in acts:
                acts.remove(self.ig_action)

        return acts         

    def step(self, action, env=None):
        # 状态 0 下落过程中 1 更换方块 2 结束一局
        
        self.reward = 0
        self.steps += 1
        self.piecesteps += 1
        # self.curr_player = (self.curr_player+1)%2
        self.curr_player = 1 if self.curr_player==0 else 0
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

        # if self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
        #     self.fallpiece['y'] +=1

        fallpiece_y = self.fallpiece['y']

        self.fallpiece_status.append(self.get_fallpiece_board())

        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.tetromino.addtoboard(self.board,self.fallpiece)
            self.reward = self.tetromino.removecompleteline(self.board) 
            self.score += self.reward          
            self.level, self.fallfreq = self.tetromino.calculate(self.score)   
            fallpiece_y += self.reward

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
            self.state = 1
            self.fallpiece_status=[self.get_fallpiece_board()]          

            # print(self.limit_piece_count, self.piececount)
            if (not self.tetromino.validposition(self.board,self.fallpiece)) or \
                (self.limit_max_height>0 and (20-fallpiece_y)>self.limit_max_height):  
                self.terminal = True 
                self.state = 1
                self.reward = -1 
                # print(">>>>>", len(self.pieces_height), self.pieces_height, self.actions[-10:])
                return self.state, self.reward # 

            # if (not self.tetromino.validposition(self.board,self.fallpiece)): 
            #     self.terminal = True 
            #     self.state = 2
            #     return self.state, self.reward # 
        else:
            self.state = 0
            
        # 早期训练中，如果得分就表示游戏结束
        # if reward>0: self.terminal=True
            
        self.availables = self.get_availables()

        return self.state, self.reward

    def get_key(self,is_include_player=True):
        info = self.getBoard() + self.fallpiece_status[-1] 
        if is_include_player and self.curr_player==1:
            info = info + np.ones((self.height, self.width))
        return hash(info.data.tobytes())   
        
    # 打印
    def print2(self, add_fallpiece=False):
        info = self.getBoard()
        if add_fallpiece:
            info += self.fallpiece_status[-1]
        for y in range(self.height):
            line=""
            for x in range(self.width):
                if info[y][x]==0:
                    line=line+"  "
                else:
                    line=line+"* "
            print(line)
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
        c = self.height
        for y in range(self.height):
            for x in range(self.width):
                if self.board[x][y]!=blank:
                    c=y
                    break
            if c!=0:break            
        return self.height - c

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
    # 背景 + 下落方块位置 + 下一次的方块 = 3
    # 返回 [3, height, width]
    def current_state(self):
        board_background = self.getBoard()
        fallpiece_1 =  self.fallpiece_status[-1]
        fallpiece_2 = np.zeros((self.height, self.width))
        fallpiece_3 = np.zeros((self.height, self.width))
        fallpiece_4 = np.zeros((self.height, self.width))
        fallpiece_5 = np.zeros((self.height, self.width))

        if len(self.fallpiece_status)>1:           
            fallpiece_2 = self.fallpiece_status[-2]  
        if len(self.fallpiece_status)>2:           
            fallpiece_3 = self.fallpiece_status[-3]  
        if len(self.fallpiece_status)>3:           
            fallpiece_4 = self.fallpiece_status[-4]  
        if len(self.fallpiece_status)>4:           
            fallpiece_5 = self.fallpiece_status[-5]  

        # board_fallpiece = self.get_fallpiece_board()
        # board_nextpiece = self.get_nextpiece_borad()
        
        # if self.curr_player==0:
        #     step_state = np.ones([self.height, self.width])
        # else:
        #     step_state = np.zeros([self.height, self.width])

        state = np.stack([board_background, fallpiece_5, fallpiece_4, fallpiece_3, fallpiece_2, fallpiece_1])
        return state        

    # 交替个数也就是从空到非空算一次，边界算非空 
    # 高度从底部 20 --> 0 上部
    def getTransCount(self, board=None):
        if board==None: board = self.board

        _rowTransitions = rowTransitions(board)
        _colTransitions = colTransitions(board)
        _emptyHoles = emptyHoles(board)
        _wellNums = wellNums(board)
        return  3.2178882868487753 * _rowTransitions \
                + 9.348695305445199 * _colTransitions \
                + 7.899265427351652 * _emptyHoles \
                + 3.3855972247263626 * _wellNums; 

        transCount = 0

        # 统计一列的个数
        for x in range(self.width):
            curr_state = 1
            for y in range(self.height)[::-1]:
                state = 0 if board[x][y]==blank else 1
                if curr_state!=state:
                    transCount += self.height-y+1 
                    curr_state = state

        # 统计一行的个数
        for y in range(self.height):
            curr_state = 1
            for x in range(self.width):
                state = 0 if board[x][y]==blank else 1
                if curr_state!=state:
                    transCount += self.height-y+1 
                    curr_state = state
            if curr_state == 0: transCount += self.height-y+1  

        return transCount

    # 检测这一步是否优，如果好+1，不好-1，无法评价0
    # def checkActionisBest(self, include_fallpiece=True):
    #     board = [[0]*self.width for i in range(self.height)]
    #     for y in range(self.height):
    #         for x in range(self.width):
    #             board[y][x]=self.board[x][y]

    #     if self.fallpiece != None and include_fallpiece:
    #         piece = self.fallpiece
    #         shapedraw = pieces[piece['shape']][piece['rotation']]
    #         offset_y = 0
    #         for t in range(self.height):
    #             find=False
    #             for y in range(templatenum):
    #                 for x in range(templatenum):
    #                     if shapedraw[y][x]!=blank:
    #                         px, py = x+piece['x'], y+piece['y']+t
    #                         if py>=self.height or board[py][px]!=blank:
    #                             find=True
    #                             break
    #                 if find: break
    #             if find:
    #                 offset_y=t-1
    #                 break
            
    #         for y in range(templatenum):
    #             for x in range(templatenum):
    #                 if shapedraw[y][x]!=blank:
    #                     px, py = x+piece['x'], y+piece['y']+offset_y
    #                     if px>=0 and py>=0:
    #                         board[py][px]=shapedraw[y][x]
    #     transCount = self.getTransCount(board)
    #     # v = self.transCount - transCount
    #     # if self.state != 0: 
    #     #     self.transCount = transCount
    #     return transCount    

    # 检查游戏是否结束，如果有奖励，下棋的赢了，否则输了
    def game_end(self):
        if self.terminal:
            # if self.score>0:
            #     return True, 0 
            # else:
            return True, 1
        else:
            return False, -1 
        

    # 这里假定第一个人选择下[左移，右移，翻转，下降，无动作]，第二个人只有[下降]
    # def start_self_play(self, player, temp=1e-3):       
    #     states, mcts_probs = [], []
    #     self.reset()
    #     player.reset_player()
    #     for i in count():
    #         # temp 权重 ，return_prob 是否返回概率数据
    #         action, move_probs = player.get_action(self, temp=temp/(self.piecesteps+1), return_prob=1)
    #         # 保存数据
    #         states.append(self.current_state())
    #         mcts_probs.append(move_probs)

    #         # 前几步是乱走的
    #         # if self.piecesteps<10-self.piececount and random.random()>0.5:
    #         # 有20%是乱走的
    #         # if random.random()>0.8:
    #         #     action = random.choice(self.availables())

    #         # 执行一步
    #         self.step(action)
    #         # 如果游戏结束
    #         if self.terminal: break
    #     self.print()

    #     picece_count = self.getTransCount()
    #     print("transCount:", picece_count)
    #     # 增加最大步骤
    #     if picece_count>maxstep:
    #         states1 = states
    #         mcts_probs1 = mcts_probs
    #         winners1 = [-1.0 for i in range(len(states))]
    #         maxstep = picece_count
    #     elif picece_count==maxstep:
    #         states1 = states1 + states
    #         mcts_probs1 = mcts_probs1 + mcts_probs
    #         winners1 = winners1+ [-1.0 for i in range(len(states))]
    #     # 增加最小步骤
    #     if picece_count<minstep:
    #         states0 = states
    #         mcts_probs0 = mcts_probs
    #         winners0 = [1.0 for i in range(len(states))]
    #         minstep = picece_count
    #     # 增加最小步骤
    #     elif picece_count==minstep:
    #         states0 = states0 + states
    #         mcts_probs0 = mcts_probs0 + mcts_probs
    #         winners0 = winners0 + [1.0 for i in range(len(states))]

    #     print("minstep",minstep,"maxstep",maxstep)
    #     states = states0 + states1
    #     mcts_probs = mcts_probs0 + mcts_probs1
    #     winners = winners0 + winners1

    #     assert len(states)==len(mcts_probs)==len(winners)
    #     return -1, zip(states, mcts_probs, winners)

    # # 使用 mcts 训练，重用搜索树，并保存数据
    def start_self_play(self, player, temp=1e-3):
        # 这里下两局，按步数对比


        # self.ig_action = random.choice([None,KEY_NONE,KEY_DOWN])

        if self.limit_max_height > 0:
            limit_max_height = self.limit_max_height
        else:
            limit_max_height = random.choice([25,10,10,10])
            self.limit_max_height = limit_max_height

        # 玩几局订胜负，如果最高为5，玩4局，否则玩2局
        if limit_max_height != 10:
            game_num = 5
            player.mcts._n_playout = 32
        else:
            game_num = 2
        
        game_states, game_mcts_probs, game_masks = [],[],[] 
        game_piececount, game_score = [],[]
        print("limit_max_height", self.limit_max_height)
        for j in range(game_num):
            _states, _mcts_probs, _masks=[],[],[]
            game = copy.deepcopy(self)
            game.limit_max_height = 5
            # ig_action=random.choice([None,KEY_NONE,KEY_DOWN])
            # game.ig_action = ig_action

            for i in count():                           
                action, move_probs = player.get_action(game, temp=temp, return_prob=1) 
                _states.append(game.current_state())
                if game.curr_player==0:
                    _mcts_probs.append(move_probs)
                    _masks.append(1)
                else:
                    _mcts_probs.append(np.ones([game.actions_num])/game.actions_num)
                    _masks.append(0)

                game.step(action)
                if game.state!=0:
                    game.limit_max_height = max(game.pieces_height)+3
                    if game.limit_max_height>limit_max_height: game.limit_max_height=limit_max_height
                    print('reward:',game.reward, 'len:', len(game.pieces_height), "limit_max_height:", game.limit_max_height, "next:", game.fallpiece['shape'], game.pieces_height)

                if game.terminal:
                    break

            game_states.append(_states)
            game_mcts_probs.append(_mcts_probs)
            game_masks.append(_masks)

            game_piececount.append(game.piececount)
            game_score.append(game.score)

            game.print()

            if j>=2 and limit_max_height != 10:
                max_p = max(game_piececount)
                if game_piececount.count(max_p)==1:
                    break

        max_piececount = max(game_piececount)
        max_score = max(game_score)

        game_win = [-1 for _ in range(game_num)] 
        game_loss = [1 for _ in range(game_num)] 
        for j in range(game_num):
            if game_piececount[j]==max_piececount:
                game_win[j] = 1
            if game_score[j]>=limit_max_height//5 and game_score[j] == max_score:
                game_loss[j] = -1

        print("game_piececount",game_piececount,"game_score",game_score)
        print("win",game_win,"score",game_loss)
        
        states, mcts_probs, winers, masks = [], [], [], []
        for j in range(game_num):
            for o in game_states[j]: states.append(o)
            for o in game_masks[j]:  masks.append(o)
            for o in game_mcts_probs[j]: mcts_probs.append(o)
            for m in game_masks[j]:
                if m==1:
                    winers.append(game_win[j])
                else:
                    winers.append(game_loss[j])

        winners_z = np.array(winers)

        assert len(states)==len(mcts_probs)
        assert len(states)==len(winners_z)
        assert len(states)==len(masks)

        print("add %s to dataset"%len(winers))
        reward, piececount, agentcount = 0, 0, 0
        reward = sum(game_score)  
        piececount = sum(game_piececount)
        agentcount = game_num
    
        return reward, piececount, agentcount, zip(states, mcts_probs, winners_z, masks)

    