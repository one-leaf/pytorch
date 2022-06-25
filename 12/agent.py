from game import Tetromino, pieces, templatenum, blank 
# from pygame.locals import *
import numpy as np
import random
from collections import deque
import math

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
        # 方块的最高高度
        self.pieceheight = 0
        # 面板
        self.board = self.tetromino.getblankboard()
        # 状态： 0 下落过程中 1 更换方块 2 结束一局
        self.state = 0
        # 上一个下落方块的截图
        self.prev_fallpiece_boards=None
        # 每个方块的高度
        self.pieces_height = []     
        # 盘面的状态
        self.status = deque(maxlen=10)
        for i in range(9):
            self.status.append(np.zeros((self.height, self.width)))
        self.status.append(self.get_fallpiece_board())
        # 下一个可用步骤
        self.availables=self.get_availables()
        # 显示mcts中间过程
        self.show_mcts_process = False
        # pos
        self.pos_board = self.get_board_pos()
        # key
        self.key=0


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

        if not KEY_DOWN in acts : acts.append(KEY_NONE)

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
        if self.tetromino.validposition(self.board,self.fallpiece, ay=1):
            self.fallpiece['y'] +=1
        else:
            isFalling = False

        fallpiece_y = self.fallpiece['y']

        self.status.append(self.get_fallpiece_board() + self.getBoard())
        self.set_key()

        if not isFalling:
            self.tetromino.addtoboard(self.board,self.fallpiece)            
            self.reward = self.tetromino.removecompleteline(self.board) 
            
            self.score += self.reward
            self.pieceheight = self.getMaxHeight()          
            self.pieces_height.append(20 - fallpiece_y - self.reward)
            self.fallpiece = None

        if  env:
            env.checkforquit()
            env.render(self.board, self.score, self.level, self.fallpiece, self.nextpiece)

        if not isFalling:
            self.fallpiece = self.nextpiece
            self.nextpiece = self.tetromino.getnewpiece()
            self.piecesteps = 0
            self.piececount +=1 

            if not self.tetromino.validposition(self.board,self.fallpiece, ay=1):                  
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

    def set_key(self):
        info = self.status[-1]
        self.key = hash(info.data.tobytes())

    def get_key(self):
        return self.key

    # 打印
    def print2(self):
        info = self.status[-1]
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
        c = -1
        for y in range(self.height):
            for x in range(self.width):
                if self.board[x][y]!=blank:
                    c=y
                    break
            if c!=-1:break  
        h = 0 if c == -1 else self.height - c                          
        return h

    # 统计非空的个数
    def getNoEmptyCount(self):
        c = 0
        for y in range(self.height):
            line_c= 0
            for x in range(self.width):
                if self.board[x][y]!=blank:
                    line_c += 1
            c += line_c*(0.9**(self.height-y-1))

            # if line_c == 9: c += 1
        return c

    # 计算得分,只计算被挡住的
    def getScore(self):
        empty_count = 0 
        fill_count = 0
        for x in range(self.width):
            line_f, line_e = 0, -1
            for y in range(self.height):
                if self.board[x][y] != blank:
                    line_f += 1
                    if line_e == -1: line_e = 0
                else:
                    if line_e !=-1: line_e += 1
            empty_count += line_e
            fill_count += line_f

        if fill_count==0: return 0
        return max(-1, -1 * (empty_count/fill_count))        

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


    # 得到面板的坐标信息
    def get_board_pos(self):
        pos=[]
        size =  self.width * self.height
        for i in range(size):
            if i%2==0:
                pos.append(math.sin(i/size))
            else:
                pos.append(math.cos(i/size))
        pos = np.array(pos).reshape((self.height, self.width))
        return pos

    # 获得当前的全部特征
    # 背景 + 前2步走法 = 3
    # 返回 [3, height, width]
    def current_state(self):
        state = np.zeros((3, self.height, self.width))
        for i in range(3):
            state[-1*(i+1)]=self.status[-1*(i+1)]

        return state          


    
                
