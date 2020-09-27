from game import Tetromino, TetrominoEnv, pieces, templatenum, blank, black
import pygame
from pygame.locals import *
from itertools import count
import numpy as np

KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN  = 0, 1, 2, 3

class Agent(object):
    def __init__(self, need_draw=False):
        self.need_draw = need_draw 
        if not need_draw:
            self.tetromino = Tetromino()
        else:
            self.tetromino = TetrominoEnv()
        self.availables = [KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
        self.width = 10
        self.height = 20
        self.reset()

    def reset(self):
        self.fallpiece = self.tetromino.getnewpiece()
        self.nextpiece = self.tetromino.getnewpiece()
        self.terminal = False
        self.score = 0
        self.level = 0
        self.board = self.tetromino.getblankboard()
    
    def step(self, action):
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
                state = 2       
                return state, reward
            else: 
                state =1
        else:
            state = 0
        
        # 这里早期训练得分直接结束游戏
        if self.score>0: self.terminal = True

        return state, reward

    # 打印
    def print(self):
        print(self.getBoard())
        print("level:", self.level, "score:", self.score)

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

    def game_end(self):
        holeCount = self.getHoleCount()
        return self.terminal, holeCount/200   #self.score

    # 使用 mcts 训练，重用搜索树，并保存数据
    def start_self_play(self, player, temp=1e-3):
        # 这里下两局，按得分和步数对比
        states, mcts_probs, current_players = [], [], []
        score_1 = score_2 = 0
        holes_1 = holes_2 = 0
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
                score_1 = self.score
                break
        self.print()

        holes_1 = self.getHoleCount()

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
                score_2 = self.score
                break
        self.print()
        holes_2 = self.getHoleCount()

        # 按照棋局得分确定输赢,如果得分一样，就按谁的空洞最小，最小的优
        winners_z = np.zeros(len(current_players))
        if score_1>0:
            winners_z[np.array(current_players) == 0] = 1.0 * score_1
        else:
            winners_z[np.array(current_players) == 0] = -1 * (1 - holes_1/200) 

        if score_2>0:
            winners_z[np.array(current_players) == 1] = 1.0 * score_2
        else:
            winners_z[np.array(current_players) == 1] = -1 * (1- holes_2/200)  

        winner = -1
        # if score_2 > score_1:
        #     winner = 1
        # if score_1 > score_2:
        #     winner = 0
        # if score_1 == score_2:
        #     if holes_1>holes_2:
        #         winner = 1
        #     if holes_2>holes_1:
        #         winner = 0

        # if winner != -1:
        #     winners_z[np.array(current_players) == winner] = 1.0
        #     winners_z[np.array(current_players) != winner] = -1.0
                
        print("score_0:",score_1,"score_1:",score_2,"holes_0:",holes_1,"holes_1:",holes_2,"winner:",winner)
        return winner, zip(states, mcts_probs, winners_z)
