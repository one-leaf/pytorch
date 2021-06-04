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
        # 触底的玩家 
        self.state_player = -1        
        # 当前已经下落的方块
        self.fallpiece_height=0
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

    # 概率的索引位置转action
    def position_to_action(self, position):
        return ACTIONS[position]

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

        acts=[KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_NONE]

        if not self.tetromino.validposition(self.board,self.fallpiece,ax = -1):
            acts.remove(KEY_LEFT)
        if not self.tetromino.validposition(self.board,self.fallpiece,ax = 1):
            acts.remove(KEY_RIGHT)   
        # if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
        #     acts.remove(KEY_DOWN)

        # 只允许前面旋转
        if self.piecesteps>len(pieces[self.fallpiece['shape']]):
            acts.remove(KEY_ROTATION)
        else:
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
        self.ig_action = action
        
        self.reward = 0
        self.steps += 1
        self.piecesteps += 1
        self.curr_player = (self.curr_player+1)%2

        self.level, self.fallfreq = self.tetromino.calculate(self.score)

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
        self.fallpiece_status.append(self.get_fallpiece_board())

        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.tetromino.addtoboard(self.board,self.fallpiece)
            self.reward = self.tetromino.removecompleteline(self.board) 
            self.score += self.reward          
            self.level, self.fallfreq = self.tetromino.calculate(self.score)   
            self.fallpiece_height = landingHeight(self.fallpiece)
            self.pieces_height.append(self.fallpiece_height)
            self.fallpiece = None

        if  env:
            env.checkforquit()
            env.render(self.board, self.score, self.level, self.fallpiece, self.nextpiece)

        if self.fallpiece == None:
            self.fallpiece = self.nextpiece
            self.nextpiece = self.tetromino.getnewpiece()
            self.piecesteps = 0
            self.piececount +=1 

            if not self.tetromino.validposition(self.board,self.fallpiece) or (self.limit_piece_count>0 and self.piececount>=self.limit_piece_count):  
                self.terminal = True 
                self.state = 2       
                return self.state, self.reward # 
            else: 
                self.state = 1

            self.state_player = self.curr_player
            self.curr_player = 0  
            self.fallpiece_status=[self.get_fallpiece_board()]          
        else:
            self.state = 0
        
        # 早期训练中，如果得分就表示游戏结束
        # if reward>0: self.terminal=True

        self.availables = self.get_availables()

        return self.state, self.reward

    def get_key(self, include_curr_player=True):
        info = self.getBoard() + self.fallpiece_status[-1]
        key = [0 for v in range(self.height*self.width)]
        for x in range(self.height):
            for y in range(self.width):
                if info[x][y]==0:
                    key[x*self.width+y]='0'
                else:
                    key[x*self.width+y]='1'
        if include_curr_player:
            key.insert(0, str(self.curr_player))
        key3 = int("".join(key),2)
        return hash(key3)

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
        c = 0
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
        board_fallpiece =  self.fallpiece_status[-1]
        if len(self.fallpiece_status)>2:           
            board_fallpiece_prev = self.fallpiece_status[-3]  
        else:
            board_fallpiece_prev = np.zeros((self.height, self.width))

        # board_fallpiece = self.get_fallpiece_board()
        # board_nextpiece = self.get_nextpiece_borad()
        
        # if self.curr_player==0:
        #     step_state = np.ones([self.height, self.width])
        # else:
        #     step_state = np.zeros([self.height, self.width])

        state = np.stack([board_background, board_fallpiece_prev, board_fallpiece])
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

    def game_end(self):
        return self.terminal, self.score

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
        states, mcts_probs, winers = [], [], []

        game0 = copy.deepcopy(self)
        game1 = copy.deepcopy(self)

        game0_states,game1_states,game0_mcts_probs,game1_mcts_probs,game0_players,game1_players=[],[],[],[],[],[]

        train_pieces_count = random.randint(2,5)  
        print("max pieces count:",train_pieces_count)
        player.reset_player()
        # game0.limit_piece_count = train_pieces_count

        for i in count():            
            # 只保留有效的步数
            # if game0.piecesteps<ig_steps:
            #     if game0.curr_player==0:
            #         action = random.choice([KEY_ROTATION, KEY_LEFT, KEY_RIGHT])
            #     else:
            #         action = KEY_DOWN
            #     game0.step(action)
            #     if game0.terminal or game0.piececount>=train_pieces_count: 
            #         game0.terminal = True
            #         break
            #     continue
               
            action, move_probs = player.get_action(game0, temp=temp, return_prob=1) 
            if game0.curr_player==0:
                game0_states.append(game0.current_state())
                game0_players.append(game0.curr_player)
                game0_mcts_probs.append(move_probs)

            game0.step(action)
            # game0.print2(True)
            if game0.terminal or game0.piececount>=train_pieces_count: 
                break

        player.reset_player()
        # game1.limit_piece_count = train_pieces_count
        for i in count():
            # if game1.piecesteps<ig_steps:
            #     if game1.curr_player==0:
            #         action = random.choice([KEY_ROTATION, KEY_LEFT, KEY_RIGHT])
            #     else:
            #         action = KEY_DOWN
            #     game1.step(action)
            #     if game1.terminal or game1.piececount>=train_pieces_count:
            #         game1.terminal = True
            #         break
            #     continue
            # 只保留有效的步数

            action, move_probs = player.get_action(game1, temp=temp, return_prob=1)
            if game1.curr_player==0:
                game1_states.append(game1.current_state())
                game1_players.append(game1.curr_player)
                game1_mcts_probs.append(move_probs)
    
            game1.step(action)
            # game1.print2(True)            
            if game1.terminal or game1.piececount>=train_pieces_count: 
                break

        game0.print()
        game1.print()

        #game0_exscore = -1 * game0.getMaxHeight()
        #game1_exscore = -1 * game1.getMaxHeight()

        game0_exscore = -1 * game0.getTransCount()
        game1_exscore = -1 * game1.getTransCount()
            
        print("game0_exscore:",game0_exscore,"game1_exscore:",game1_exscore)
        # 如果有输赢，则直接出结果，如果相同，继续下一轮，直到出结果为止
        game0_win, game1_win = 0, 0

        if game0_exscore>game1_exscore:
            game0_win, game1_win  = 1, -1

        if game0_exscore<game1_exscore:
            game0_win, game1_win  = -1, 1

        winers = []

        for i in game0_players:
            if i==0:
                winers.append(game0_win)
            else:
                winers.append(game0_win*-1)

        for i in game1_players:
            if i==0:
                winers.append(game1_win)
            else:
                winers.append(game1_win*-1)

        for o in game0_states: states.append(o)
        for o in game1_states: states.append(o)
        for o in game0_mcts_probs: mcts_probs.append(o)
        for o in game1_mcts_probs: mcts_probs.append(o)

        game0_states,game1_states,game0_mcts_probs,game1_mcts_probs,game0_wins,game1_wins=[],[],[],[],[],[]
        winners_z = np.array(winers)

        assert len(states)==len(mcts_probs)
        assert len(states)==len(winners_z)

        # winners_z = np.zeros(len(winers))
        # winners_z[np.array(winers) == 1] = 1.0
        # winners_z[np.array(winers) == -1] = -1.0
        # print(states[-1])
        # print(mcts_probs[-1])
        # print(winners_z[-1])

        print("add %s to dataset"%len(winers))
        return -1, zip(states, mcts_probs, winners_z)

    # # 使用 mcts 训练，重用搜索树，并保存数据
    # def start_self_play3(self, player, temp=1e-3):
    #     # 这里下两局，按得分和步数对比
    #     # 这样会有一个问题，导致+分比-分多，导致mcts会集中到最初和最后的步骤
    #     states, mcts_probs, current_players = [], [], []
    #     # 当方块到了这个就终止游戏
    #     max_height = 15
    #     tetromino = copy.deepcopy(self.tetromino)
    #     # 训练方块数
    #     self.reset()
    #     for i in count():
    #         # temp 权重 ，return_prob 是否返回概率数据
    #         action, move_probs = player.get_action(self, temp=temp, return_prob=1)
    #         # 保存数据
    #         states.append(self.current_state())
    #         mcts_probs.append(move_probs)
    #         current_players.append(0)
    #         # 执行一步
    #         self.step(action)
    #         # 如果游戏结束
    #         if self.terminal: break
    #         if self.state!=0 and self.getMaxHeight()>=max_height: break
    #     self.print()
    #     score0 = self.score
    #     steps0 = 200-len(self.tetromino.nextpiece)

    #     self.tetromino=tetromino
    #     self.reset()
    #     for i in count():
    #         # temp 权重 ，return_prob 是否返回概率数据
    #         action, move_probs = player.get_action(self, temp=temp, return_prob=1)
    #         # 保存数据
    #         states.append(self.current_state())
    #         mcts_probs.append(move_probs)
    #         current_players.append(1)
    #         # 执行一步
    #         self.step(action)
    #         # 如果游戏结束
    #         if self.terminal: break
    #         if self.state!=0 and self.getMaxHeight()>=max_height: break
    #     self.print()
    #     score1 = self.score
    #     steps1 = 200-len(self.tetromino.nextpiece)

    #     winner = -1
    #     winners_z = np.zeros(len(current_players))
        
    #     # 如果有奖励，按奖励大的赢,否则全部赢；如果没有奖励则按步数，谁的步数多谁赢
    #     # if score0>0 or score1>0:
    #     #     if score0!=score1:
    #     #         winner = 0 if score0>score1 else 1
    #     # else:
    #     # 至少大于1个方块的相差，如果只有相差1个方块，可以认为是平局
    #     if abs(steps0-steps1)>1:
    #         winner = 0 if steps0>steps1 else 1         

    #     if winner in [0, 1]:
    #         winners_z[np.array(current_players) == winner] = 1.0
    #         winners_z[np.array(current_players) != winner] = -1.0
    #     else:
    #         # 如果是平局，作为负分补偿，全部给负分
    #         winners_z[:] = -1.0

    #     print("winner",winner,"step0",steps0,"step1",steps1)
    #     return winner, zip(states, mcts_probs, winners_z)

    # def start_play(self, player, env):
    #     while True:
    #         action = player.get_action(self)
    #         self.step(action, env)
    #         if self.terminal:
    #             break


    # # 使用 mcts 训练，重用搜索树，并保存数据
    # def start_self_play4(self, player, temp=1e-3):
    #     # 这里下5局，取最好和最差的按得分和步数对比
    #     # 这样会有一个问题，导致+分比-分多，导致mcts会集中到最初和最后的步骤
    #     # 当方块到了这个就终止游戏
    #     max_height = 0
    #     states0,states1,mcts_probs0,mcts_probs1,winners0,winners1=None,None,None,None,None,None
    #     minstep = 999999999
    #     maxstep = 0
    #     tetromino = self.tetromino
    #     # 必须要找到相差2个方块以上的局面
    #     step = 0
    #     while maxstep-minstep<2:
    #         step += 1
    #         if step>10 and maxstep-minstep>1 : break
    #         states, mcts_probs = [], []
    #         self.tetromino=copy.deepcopy(tetromino)
    #         self.reset()
    #         player.reset_player()
    #         for i in count():
    #             # temp 权重 ，return_prob 是否返回概率数据
    #             action, move_probs = player.get_action(self, temp=temp*2/(self.piecesteps+1), return_prob=1)
    #             # 保存数据
    #             states.append(self.current_state())
    #             mcts_probs.append(move_probs)
    #             # 执行一步
    #             self.step(action)
    #             # 如果游戏结束
    #             if self.terminal: break
    #             if self.state!=0 and max_height>0 and self.getMaxHeight()>=max_height: break
    #             # if self.state!=0 : self.print()

    #         self.print()
    #         picece_count = self.piececount
    #         # 增加最大步骤
    #         if picece_count>maxstep:
    #             states1 = states
    #             mcts_probs1 = mcts_probs
    #             winners1 = [1.0 for i in range(len(states))]
    #             maxstep = picece_count
    #         elif picece_count==maxstep:
    #             states1 = states1 + states
    #             mcts_probs1 = mcts_probs1 + mcts_probs
    #             winners1 = winners1+ [1.0 for i in range(len(states))]
    #         # 增加最小步骤
    #         if picece_count<minstep:
    #             states0 = states
    #             mcts_probs0 = mcts_probs
    #             winners0 = [-1.0 for i in range(len(states))]
    #             minstep = picece_count
    #         # 增加最小步骤
    #         elif picece_count==minstep:
    #             states0 = states0 + states
    #             mcts_probs0 = mcts_probs0 + mcts_probs
    #             winners0 = winners0 + [-1.0 for i in range(len(states))]

    #     print("minstep",minstep,"maxstep",maxstep)
    #     states = states0 + states1
    #     mcts_probs = mcts_probs0 + mcts_probs1
    #     winners = winners0 + winners1

    #     assert len(states)==len(mcts_probs)==len(winners)
    #     return -1, zip(states, mcts_probs, winners)

    # def start_self_play(self, player, temp=1e-3):
    #     # 同时放X个方块，谁的熵最小，谁赢
    #     max_picece_count = random.randint(3,6)  

    #     states0,states1,mcts_probs0,mcts_probs1,winners0,winners1=None,None,None,None,None,None
    #     tetromino = self.tetromino
    #     minstep = 999999999
    #     maxstep = 0
    #     while maxstep-minstep<2:
    #         states, mcts_probs = [], []
    #         self.tetromino=copy.deepcopy(tetromino)
    #         self.reset()
    #         player.reset_player()
    #         for i in count():
    #             # temp 权重 ，return_prob 是否返回概率数据
    #             action, move_probs = player.get_action(self, temp=temp/(self.piecesteps+1), return_prob=1)
    #             # 保存数据
    #             states.append(self.current_state())
    #             mcts_probs.append(move_probs)

    #             # 前几步是乱走的
    #             # if self.piecesteps<10-self.piececount and random.random()>0.5:
    #             # 有20%是乱走的
    #             if random.random()>0.8:
    #                 action = random.choice(self.availables())

    #             # 执行一步
    #             self.step(action)
    #             # 如果游戏结束
    #             if self.terminal: break
    #             if self.state!=0 and self.piececount>=max_picece_count: break
    #         self.print()
    #         picece_count = self.getTransCount()
    #         print("transCount:", picece_count)
    #         # 增加最大步骤
    #         if picece_count>maxstep:
    #             states1 = states
    #             mcts_probs1 = mcts_probs
    #             winners1 = [-1.0 for i in range(len(states))]
    #             maxstep = picece_count
    #         elif picece_count==maxstep:
    #             states1 = states1 + states
    #             mcts_probs1 = mcts_probs1 + mcts_probs
    #             winners1 = winners1+ [-1.0 for i in range(len(states))]
    #         # 增加最小步骤
    #         if picece_count<minstep:
    #             states0 = states
    #             mcts_probs0 = mcts_probs
    #             winners0 = [1.0 for i in range(len(states))]
    #             minstep = picece_count
    #         # 增加最小步骤
    #         elif picece_count==minstep:
    #             states0 = states0 + states
    #             mcts_probs0 = mcts_probs0 + mcts_probs
    #             winners0 = winners0 + [1.0 for i in range(len(states))]

    #     print("minstep",minstep,"maxstep",maxstep)
    #     states = states0 + states1
    #     mcts_probs = mcts_probs0 + mcts_probs1
    #     winners = winners0 + winners1

    #     assert len(states)==len(mcts_probs)==len(winners)
    #     return -1, zip(states, mcts_probs, winners)