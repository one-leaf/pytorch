from game import Tetromino, TetrominoEnv, pieces, templatenum, blank, black
import pygame
from pygame.locals import *
from itertools import count
import numpy as np
import copy

KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN  = 0, 1, 2, 3

class Agent(object):
    def __init__(self):
        self.tetromino = Tetromino(isRandomNextPiece=False)
        self.width = 10
        self.height = 20
        self.actions_num = 4
        self.reset()

    def reset(self):
        self.fallpiece = self.tetromino.getnewpiece()
        self.nextpiece = self.tetromino.getnewpiece()
        self.terminal = False
        self.score = 0
        self.level = 0
        self.steps = 0
        self.board = self.tetromino.getblankboard()
        # 变化个数，用于评价这一步的优劣
        self.transCount = self.getTransCount()
        # 状态： 0 下落过程中 1 更换方块 2 结束一局
        self.state =0

    # 获取可用步骤, 保留一个旋转始终有用
    def availables(self):
        acts=[KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
        if not self.tetromino.validposition(self.board,self.fallpiece,ax = -1):
            acts.remove(KEY_LEFT)
        if not self.tetromino.validposition(self.board,self.fallpiece,ax = 1):
            acts.remove(KEY_RIGHT)   
        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            acts.remove(KEY_DOWN)
        r = self.fallpiece['rotation']
        self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
        if not self.tetromino.validposition(self.board,self.fallpiece):
            acts.remove(KEY_ROTATION)
        self.fallpiece['rotation'] = r
        return acts         

    def step(self, action, env=None):
        # 状态 0 下落过程中 1 更换方块 2 结束一局
        reward = 0
        self.steps += 1
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

        if self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.fallpiece['y'] +=1

        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.tetromino.addtoboard(self.board,self.fallpiece)
            reward = self.tetromino.removecompleteline(self.board) 
            self.score += reward          
            self.level, self.fallfreq = self.tetromino.calculate(self.score)   
            self.fallpiece = None

        if  env:
            env.checkforquit()
            env.render(self.board, self.score, self.level, self.fallpiece, self.nextpiece)

        if self.fallpiece == None:
            self.fallpiece = self.nextpiece
            self.nextpiece = self.tetromino.getnewpiece()
            if not self.tetromino.validposition(self.board,self.fallpiece):  
                self.terminal = True 
                self.state = 2       
                return self.state, reward
            else: 
                self.state =1
        else:
            self.state = 0
        
        # 早期训练中，如果得分就表示游戏结束
        # if reward>0: self.terminal=True

        return self.state, reward

    # 打印
    def print(self, add_fallpiece=False):
        info = self.getBoard()
        if add_fallpiece:
            info += self.get_fallpiece_board()
        for y in range(self.height):
            line=""
            for x in range(self.width):
                if info[y][x]==0:
                    line=line+"  "
                else:
                    line=line+"* "
            print(line)
        print("level:", self.level, "score:", self.score, "steps:", self.steps)

    # 统计空洞数量
    def getHoleCount(self):
        c = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.board[x][y]==blank:
                    c += 1
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
    def current_state(self):
        board = self.getBoard()
        board_1 = self.get_fallpiece_board()
        board_2 = self.get_nextpiece_borad()
        state = np.stack([board,board_1,board_2])
        return state        

    # 交替个数也就是从空到非空算一次，边界算非空 
    def getTransCount(self, board=None):
        if board==None: board = self.board
        height = len(board)
        width = len(board[0])

        transCount = 0

        # 由于高度有额外的影响力，所以增加了高度的权重
        for x in range(width):
            curr_state = 1
            for y in range(height)[::-1]:
                state = 0 if board[y][x]==blank else 1
                if curr_state!=state:
                    transCount += 1 + (height-y-1)/height
                    curr_state = state

        for y in range(height):
            curr_state = 1
            for x in range(width):
                state = 0 if board[y][x]==blank else 1
                if curr_state!=state:
                    transCount += 1
                    curr_state = state

        return transCount

    # 检测这一步是否优，如果好+1，不好-1，无法评价0
    def checkActionisBest(self, include_fallpiece=True):
        board = [[0]*self.width for i in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                board[y][x]=self.board[x][y]

        if self.fallpiece != None and include_fallpiece:
            piece = self.fallpiece
            shapedraw = pieces[piece['shape']][piece['rotation']]
            offset_y = 0
            for t in range(self.height):
                find=False
                for y in range(templatenum):
                    for x in range(templatenum):
                        if shapedraw[y][x]!=blank:
                            px, py = x+piece['x'], y+piece['y']+t
                            if py>=self.height or board[py][px]!=blank:
                                find=True
                                break
                    if find: break
                if find:
                    offset_y=t-1
                    break
            
            for y in range(templatenum):
                for x in range(templatenum):
                    if shapedraw[y][x]!=blank:
                        px, py = x+piece['x'], y+piece['y']+offset_y
                        if px>=0 and py>=0:
                            board[py][px]=shapedraw[y][x]
        transCount = self.getTransCount(board)
        v = self.transCount - transCount
        if self.state != 0: 
            self.transCount = transCount
        # if v>0: return 1.0
        # if v<0: return -1.0
        # if badHoleCount ==0 and v==0: return 0
        # if badHoleCount ==0 and v!=0: return v
        return v, transCount, self.transCount #(v/badHoleCount)     

    def game_end(self):
        score = 0
        if self.terminal:
            if self.score>0:
                score = 1.0 #* self.score
            else:
                holeCount = self.getHoleCount()
                score = -1.0 * (holeCount/200)
        return self.terminal, score   #self.score

    # 使用 mcts 训练，重用搜索树，并保存数据
    def start_self_play(self, player, temp=1e-3):
        # 这里下两局，按得分和步数对比
        states, mcts_probs, winers = [], [], []

        game0 = copy.deepcopy(self)
        game1 = copy.deepcopy(self)

        game0_states,game1_states,game0_mcts_probs,game1_mcts_probs,game0_wins,game1_wins=[],[],[],[],[],[]
        while not (game0.terminal or game1.terminal):

            # 一个方块一个方块的训练
            for i in count():
                action, move_probs = player.get_action(game0, temp=temp, return_prob=1)
                game0_states.append(game0.current_state())
                game0_mcts_probs.append(move_probs)
                game0.step(action)
                if game0.state!=0: break

            for i in count():
                action, move_probs = player.get_action(game1, temp=temp, return_prob=1)
                game1_states.append(game1.current_state())
                game1_mcts_probs.append(move_probs)
                game1.step(action)
                if game1.state!=0: break

            # game0.print()
            # game1.print()

            game0_transCount = game0.getTransCount()
            game1_transCount = game1.getTransCount()
            
            print("game0_transCount:",game0_transCount,"game1_transCount:",game1_transCount)
            # 如果有输赢，则直接出结果，如果相同，继续下一轮，直到出结果为止
            if game0_transCount != game1_transCount:
                game0_win, game1_win = -1, -1
                # 比谁的交换次数少
                if game0_transCount>game1_transCount:
                    game0_win, game1_win  = 0, 1
                    game0 = copy.deepcopy(game1)

                if game0_transCount<game1_transCount:
                    game0_win, game1_win  = 1, 0
                    game1 = copy.deepcopy(game0)

                for i in range(len(game0_states)):
                    game0_wins.append(game0_win)
                for i in range(len(game1_states)):
                    game1_wins.append(game1_win)

                for o in game0_states: states.append(o)
                for o in game1_states: states.append(o)
                for o in game0_mcts_probs: mcts_probs.append(o)
                for o in game1_mcts_probs: mcts_probs.append(o)
                for o in game0_wins: winers.append(o)
                for o in game1_wins: winers.append(o)

                game0_states,game1_states,game0_mcts_probs,game1_mcts_probs,game0_wins,game1_wins=[],[],[],[],[],[]
                assert len(states)==len(mcts_probs)
                assert len(states)==len(winers)
        winners_z = np.zeros(len(winers))
        winners_z[np.array(winers) == 1] = 1.0
        winners_z[np.array(winers) == 0] = -1.0
        game1.print()
        print("add %s to dataset"%len(winers))
        return -1, zip(states, mcts_probs, winners_z)



    # 使用 mcts 训练，重用搜索树，并保存数据
    def start_self_play2(self, player, temp=1e-3):
        # 这里下两局，按得分和步数对比
        states, mcts_probs, current_players = [], [], []

        tetromino = copy.deepcopy(self.tetromino)

        self.reset()
        for i in count():
            # temp 权重 ，return_prob 是否返回概率数据
            action, move_probs = player.get_action(self, temp=temp, return_prob=1)
            # 保存数据
            states.append(self.current_state())
            mcts_probs.append(move_probs)
            current_players.append(0)
            # 执行一步
            self.step(action)
            # 如果游戏结束
            if self.terminal:
                break
        self.print()
        score0 = self.score
        transCount0 = self.getTransCount()

        self.tetromino=tetromino
        self.reset()
        for i in count():
            # temp 权重 ，return_prob 是否返回概率数据
            action, move_probs = player.get_action(self, temp=temp, return_prob=1)
            # 保存数据
            states.append(self.current_state())
            mcts_probs.append(move_probs)
            current_players.append(1)
            # 执行一步
            self.step(action)
            # 如果游戏结束
            if self.terminal:
                break
        self.print()
        score1 = self.score
        transCount1 = self.getTransCount()

        winner = -1
        winners_z = np.zeros(len(current_players))
        
        # 如果有奖励，则全部赢
        if score0>0 and score0>score1: 
            winner = 0
        if score1>0 and score1>score0: 
            winner = 1

        # 如果双方都有奖励，空洞少的赢,如果平局，全部奖励
        if score0>0 and score1>0 and score0==score1:
            if abs(transCount0-transCount1)<5:
                winners_z[:] = 1.0
            else:
                winner = 0 if transCount0<transCount1 else 1

        # 如果双方都没有奖励就是平局，因为很难消除

        # 如果没有奖励，则空洞少的赢
        if score0==0 and score1==0:
            if transCount0<transCount1:
                winner = 0
            if transCount0>transCount1: 
                winner = 1   

        if winner != -1:
            winners_z[np.array(current_players) == winner] = 1.0
            winners_z[np.array(current_players) != winner] = -1.0
        print("winner:",winner,"transCount0:",transCount0,"transCount1",transCount1)
        return winner, zip(states, mcts_probs, winners_z)

    def start_play(self, player, env):
        while True:
            action = player.get_action(self)
            self.step(action, env)
            if self.terminal:
                break