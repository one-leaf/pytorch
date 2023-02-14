from os import stat
from game import Tetromino, pieces, templatenum, blank 
# from pygame.locals import *
import numpy as np
# import random
# from collections import deque
import math
import copy


KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN = 0, 1, 2, 3
ACTIONS = [KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
ACTIONS_NAME = ["O","L","R","D"]
class Agent(object):
    def __init__(self, isRandomNextPiece=False, max_height=20):
        self.width = 10
        self.height = 20
        self.actions_num = len(ACTIONS)    
        self.isRandomNextPiece = isRandomNextPiece       
        self.max_height = max_height
        self.id = 0
        self.reset()


    def reset(self):
        self.tetromino = Tetromino(isRandomNextPiece=self.isRandomNextPiece)
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
        # 方块的平均高度
        self.pieceheight = 0
        # 方块空洞数量
        self.emptyCount = 0
        # 方块最大高度差
        self.heightDiff = 0
        # 方块高度标准差
        self.heightStd = 0
        # 当前方块的高度
        self.fallpieceheight = 0
        # 面板
        self.board = self.tetromino.getblankboard()
        # 状态： 0 下落过程中 1 更换方块 2 结束一局
        self.state = 0
        # 每个方块的高度
        self.pieces_height = []     
        # 当前所有动作
        self.actions=[]
        # 盘面的状态
        self.status = [] #deque(maxlen=10)
        _board = np.zeros((self.height, self.width))
        for i in range(8):
            self.status.append(_board)
        self.add_status()
        # self.status.append(self.get_fallpiece_board()+self.getBoard())
        # 下一个可用步骤
        self.availables=self.get_availables()
        # 显示mcts中间过程
        self.show_mcts_process = False
        # key
        self.set_key()

    def add_status(self):
        self.status.append(self.get_fallpiece_board()+self.getBoard())
        del self.status[1]
        if self.state!=0:
            self.status[0]=self.getBoard()+self.get_nextpiece_borad()

        # if self.state!=0:
        #     self.status[0]=np.zeros((self.height,self.width))
        #     sPc = bin(self.piececount)[2:]
        #     for i, c in enumerate(sPc[::-1]):
        #         if c=='1':
        #             self.status[0][i]=np.ones((self.width))
        #         else:
        #             self.status[0][i]=np.zeros((self.width))


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

        if not self.tetromino.validposition(self.board, self.fallpiece, ax = -1):
            acts.remove(KEY_LEFT)
        if not self.tetromino.validposition(self.board, self.fallpiece, ax = 1):
            acts.remove(KEY_RIGHT)   
        if not self.tetromino.validposition(self.board, self.fallpiece, ay = 1):
            acts.remove(KEY_DOWN)

        if self.fallpiece['shape']=="o":
            acts.remove(KEY_ROTATION)
        else:
            r = self.fallpiece['rotation']
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.tetromino.validposition(self.board,self.fallpiece):
                acts.remove(KEY_ROTATION)
            self.fallpiece['rotation'] = r

        if not KEY_DOWN in acts : acts.append(KEY_DOWN)

        # random.shuffle(acts)
        
        return acts         

    def step(self, action, env=None):
        # 状态 0 下落过程中 1 更换方块 2 结束一局
        
        self.reward = 0
        self.steps += 1
        self.piecesteps += 1
        self.level, self.fallfreq = self.tetromino.calculate(self.score)
        
        self.actions.append(action)

        if action == KEY_LEFT and self.tetromino.validposition(self.board, self.fallpiece, ax=-1):
            self.fallpiece['x'] -= 1

        if action == KEY_RIGHT and self.tetromino.validposition(self.board, self.fallpiece, ax=1):
            self.fallpiece['x'] += 1  

        if (action == KEY_DOWN) and self.tetromino.validposition(self.board, self.fallpiece, ay=1):
            self.fallpiece['y'] += 1  

        if action == KEY_ROTATION:
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.tetromino.validposition(self.board,self.fallpiece):
                self.fallpiece['rotation'] = (self.fallpiece['rotation'] - 1) % len(pieces[self.fallpiece['shape']])

        isFalling=True
        if self.tetromino.validposition(self.board, self.fallpiece, ay=1):
            self.fallpiece['y'] += 1
            if not self.tetromino.validposition(self.board, self.fallpiece, ay=1):
                isFalling = False
        else:
            isFalling = False

        self.fallpieceheight = self.fallpiece['y']

        if not isFalling:
            self.tetromino.addtoboard(self.board, self.fallpiece)            
            self.reward = self.tetromino.removecompleteline(self.board) 
            
            self.score += self.reward
            self.pieceheight = self.getAvgHeight()    
            self.emptyCount = self.getEmptyCount()   
            self.heightDiff = self.getHeightDiff()
            self.heightStd = self.getHeightStd()   
            self.pieces_height.append(20 - self.fallpieceheight - self.reward)
            self.state = 1
            self.piecesteps = 0
            self.piececount += 1 

            self.fallpiece = self.nextpiece
            self.nextpiece = self.tetromino.getnewpiece()
            self.availables = [KEY_DOWN]
        else:
            self.state = 0

        self.add_status()
        self.set_key()

        if  env:
            env.checkforquit()
            env.render(self.board, self.score, self.level, self.fallpiece, self.nextpiece)

        if not isFalling and (not self.tetromino.validposition(self.board, self.fallpiece, ay=1) or self.pieceheight>self.max_height):                  
            self.terminal = True 
            self.state = 2
            return self.state, self.reward 
        
        self.availables = self.get_availables()    

        return self.state, self.reward

    def set_key(self):
        info = self.status[-1]
        self.key = hash(info.data.tobytes())+self.id
        # chars="abcdefghijklmnopqrstuvwxyz" 
        # key = ""
        # for x in range(self.width):
        #     h = "a"
        #     for y in range(self.height):
        #         if self.board[x][y]!=blank:
        #             h = chars[y]
        #             break
        #     key = key + h
        # r = self.fallpiece['rotation']
        # key = key + chars[r]
        # x = self.fallpiece['x']
        # key = key + chars[x]
        # y = self.fallpiece['y']
        # key = key + chars[y]
        # key = key + self.fallpiece['shape']
        # self.key = key

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
        print("score:", self.score, "steps:", self.steps,"piececount:", self.piececount, "pieceheight:", self.pieceheight)

    def print(self):
        board = copy.deepcopy(self.board)
        for x in range(templatenum):
            for y in range(templatenum):
                w = x + self.fallpiece['x']
                h = y + self.fallpiece['y']
                if pieces[self.fallpiece['shape']][self.fallpiece['rotation']][y][x]!=blank:
                    if w>=0 and w<self.width and h>=0 and h<self.height:
                        board[w][h] = self.fallpiece['color']

        for y in range(self.height):
            line="| "
            for x in range(self.width):
                if board[x][y]==blank:
                    line=line+"  "
                else:
                    line=line+str(board[x][y])+" "
            print(line)
        print(" "+" -"*self.width)
        print("score:", self.score, "piececount:", self.piececount, "emptyCount:", self.emptyCount, "pieceheight:", self.pieceheight, "heightDiff:", self.heightDiff, "getHeightStd:", self.heightStd)

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

    # 统计当前平均高度
    def getAvgHeight(self, std=False):
        h = np.zeros((self.width))
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x][y]!=blank:
                    h[x]=self.height-y
                    break   
        h_mean = np.mean(h)
        return h_mean

    # 统计高度标准差
    def getHeightStd(self):
        h = np.zeros((self.width+2))
        for x in range(self.width):
            for y in range(self.height):
                if self.board[x][y]!=blank:
                    h[x+1]=self.height-y
                    break
        h[0]=h[2]
        h[-1]=h[-3]
        v = np.zeros((self.width))
        for x in range(self.width):
            v[x] = max(abs(h[x]-h[x+1]) , abs(h[x+2]-h[x+1]))
        return np.std(v)

    # 统计数据相邻差值
    def getHeightDiff(self):
        h = np.zeros((self.width))
        for x in range(self.width):            
            for y in range(self.height):
                if self.board[x][y]!=blank:
                    h[x]=self.height-y
                    break
        v = [0]
        if h[1]-h[0]>2: v.append((h[1]-h[0]))
        if h[2]-h[1]>2: v.append((h[2]-h[1]))
        if h[3]-h[2]>2: v.append((h[3]-h[2]))
        if h[4]-h[3]>2: v.append((h[4]-h[3]))
        if abs(h[4]-h[5])>2: v.append(abs(h[4]-h[5]))
        if h[5]-h[6]>2: v.append((h[5]-h[6]))
        if h[6]-h[7]>2: v.append((h[6]-h[7]))
        if h[7]-h[8]>2: v.append((h[7]-h[8]))
        if h[8]-h[9]>2: v.append((h[8]-h[9]))
        return max(v)

    # 统计空洞的个数
    def getEmptyCount(self):
        # 每一列的空洞最高点
        c = np.zeros((self.width))
        h = np.zeros((self.width+2))
        for x in range(self.width):
            find_block=False
            for y in range(self.height):
                if find_block==False and self.board[x][y]!=blank:
                    find_block = True
                    h[x+1]=self.height-y
                if find_block and self.board[x][y]==blank: 
                    c[x] += 1
        h[0]=h[2]
        h[-1]=h[-3]
        # 加上夹壁
        for x in range(self.width):
            _c=max(h[x]-h[x+1],h[x+2]-h[x+1]) #山峰高度
            if _c>=3:
                c[x] += (_c-3)

        # 加上高度差
        _c = (max(h) - min(h))-3 
        _c = _c if _c >0 else 0 
        return sum(c) + _c

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
                            board[py][px]=-1
        # else:
        #     print("fallpiece is None")
        return board

    # 获得nextboard
    # def get_nextpiece_borad(self):
    #     board=np.zeros((self.height, self.width))
    #     shape = self.nextpiece['shape']
    #     idx = list(pieces.keys()).index(shape)
    #     board[0][idx]=1
    #     return board

    # # 获得待下落方块的信息
    def get_nextpiece_borad(self):
        board=np.zeros((self.height, self.width))
        off = int(self.width)//2-int(templatenum//2)
        if self.nextpiece != None:
            piece = self.nextpiece  
            shapedraw = pieces[piece['shape']][piece['rotation']]
            for x in range(templatenum):
                for y in range(templatenum):
                    if shapedraw[y][x]!=blank:
                        board[y][x+off]=-1
        return board


    # 获得当前的全部特征
    ## 背景 + 前2步走法 = 3
    # 背景 + 最后一步 + 合并后旋转90度
    # 返回 [3, height, width]
    def current_state(self):
        return np.array(self.status)

        # state = np.zeros((7, self.height, self.width))

        # # bg = self.status[-1][1] + self.status[-1][2] 
        # # bg_rot = np.rot90(bg).reshape(self.height, self.width)
        # # state[0] = bg_rot 
        # # state[0] = self.status[-2][1]
        # state[0] = self.status[-1][0]
        # state[1] = self.status[-1][1]
        # state[2] = self.status[-1][2]
        # # for i in range(3):
        # #     state[-1*(i+1)]=self.status[-1*(i+1)]

        # return state          


    
                
