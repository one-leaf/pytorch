from game import Tetromino, pieces, templatenum, blank, black
import pygame
from pygame.locals import *
from itertools import count

import torch

KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN  = 0, 1, 2, 3

class Agent(object):
    def __init__(self):
        self.tetromino = Tetromino()
        self.availables = [KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
        self.actions_num = 4
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

    def current_state(self):
        board = self.getBoard()
        board_1 = self.get_fallpiece_board()
        board_2 = self.get_nextpiece_borad()
        state = torch.stack([board,board_1,board_2])
        return state        

    # 获得当前局面信息
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

    # 获得下落方块的信息
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

    # 获得待下落方块的信息
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

    # 使用 mcts 训练，重用搜索树，并保存数据
    def start_self_play(self, player, temp=1e-3):
        self.reset()
        states, mcts_probs, current_players = [], [], []
        for i in count():
            # temp 权重 ，return_prob 是否返回概率数据
            action, move_probs = player.get_action(self.game, temp=temp, return_prob=1)
            # store the data
            states.append(self.current_state())
            # print(action)
            # print(move_probs.reshape(self.size,self.size))
            # print(states[-1])
            mcts_probs.append(move_probs)
            # perform a move
            self.game.step(action)
            if self.is_shown:
                self.env.render()
            end, winner = self.game.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if self.is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
