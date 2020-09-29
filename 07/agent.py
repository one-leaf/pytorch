from game import Tetromino, TetrominoEnv, pieces, templatenum, blank, black
import pygame
from pygame.locals import *
from itertools import count
import numpy as np
import copy

KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN  = 0, 1, 2, 3

class Agent(object):
    def __init__(self, need_draw=False):
        self.need_draw = need_draw 
        if not need_draw:
            self.tetromino = Tetromino(isRandomNextPiece=False)
        else:
            self.tetromino = TetrominoEnv()
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
        # 坏洞的个数，用于评价这一步的优劣
        self.badHoleCount = 0
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
        return acts         

    def step(self, action):
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

        if self.need_draw:
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
                self.terminal = True 
                self.state = 2       
                return self.state, reward
            else: 
                self.state =1
        else:
            self.state = 0
        
        # 早期训练中，如果得分就表示游戏结束
        if reward>0: self.terminal=True

        return self.state, reward

    # 打印
    def print(self):
        info = self.getBoard()+self.get_fallpiece_board()
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

    # 空洞个数 
    def getEmptyHolesCount(self):
        boardwidth = len(self.board)
        boardheight = len(self.board[0])
        holesCount = 0
        for x in range(boardwidth):
            find_block = False
            for y in range(boardheight):
                if self.board[x][y]!=blank:
                    find_block = True
                elif find_block:
                    holesCount += 1   
        # 别出现#
        for x in range(boardwidth):
            c = 0
            for y in range(boardheight):
                if self.board[x][y]!=blank: break
                if x == 0:
                    if self.board[x+1][y]!=blank:
                        c += 1
                    elif c > 0:
                        c += 1
                elif x == boardwidth-1:
                    if self.board[x-1][y]!=blank:
                        c += 1
                    elif c > 0:
                        c += 1
                else:
                    if self.board[x-1][y]!=blank and self.board[x+1][y]!=blank:
                        c += 1
                    elif c > 0:
                        c += 1
            if c>=2: holesCount += c*0.5
        return holesCount

    # 检测这一步是否优，如果好+1，不好-1，无法评价0
    def checkActionisBest(self):
        if self.state == 0: return 0
        badHoleCount = self.getEmptyHolesCount()
        v = self.badHoleCount - badHoleCount
        self.badHoleCount = badHoleCount
        # if v>0: return 1.0
        # if v<0: return -1.0
        return v*1.0    

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
        badHoleCount0 = self.getEmptyHolesCount()

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
        badHoleCount1 = self.getEmptyHolesCount()

        winner = -1
        winners_z = np.zeros(len(current_players))
        
        # 如果有奖励，则全部赢
        if score0>0: 
            winners_z[np.array(current_players) == 0] = 1.0
        if score1>0: 
            winners_z[np.array(current_players) == 1] = 1.0

        # 如果没有奖励，则空洞少的赢
        if score0==0 and score1==0:
            if badHoleCount0<badHoleCount1:
                winner = 0
            if badHoleCount0>badHoleCount1: 
                winner = 1   

        if winner != -1:
            winners_z[np.array(current_players) == winner] = 1.0
            winners_z[np.array(current_players) != winner] = -1.0
        return winner, zip(states, mcts_probs, winners_z)
