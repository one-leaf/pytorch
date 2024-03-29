from os import stat
from game import Tetromino, pieces, templatenum, blank 
# from pygame.locals import *
import numpy as np
import random
# from collections import deque
import math
import copy


KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN = 0, 1, 2, 3
ACTIONS = [KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
ACTIONS_NAME = ["O","L","R","D"]
class Agent():
    def __init__(self, isRandomNextPiece=False, must_reward_pieces_count=8, nextpieces=[]):
        self.width = 10
        self.height = 20
        self.actions_num = len(ACTIONS)    
        self.isRandomNextPiece = isRandomNextPiece       
        self.must_reward_piece_count = must_reward_pieces_count
        self.nextpieces = nextpieces        
        self.reset()

    def reset(self):
        self.tetromino = Tetromino(isRandomNextPiece=self.isRandomNextPiece)
        if len(self.nextpieces)>0: self.tetromino.nextpiece = self.nextpieces
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
        # 方块不能消除的行数
        self.failLines = 0        
        # 方块不能消除的最大高度
        self.failtop = 0
        # 当前方块的高度
        self.fallpieceheight = 0
        # mcts的额外奖励
        self.exreward=False
        # mcts的额外奖励因子
        self.exrewardRate=0 
        # 限制步骤
        self.limitstep=False
        # 面板
        self.board = self.tetromino.getblankboard()
        # 状态： 0 下落过程中 1 更换方块 2 结束一局
        self.state = 0
        # 每个方块的高度
        self.pieces_height = []     
        # 当前prices所有动作
        self.actions=[]
        # 最后一次得奖的方块序号
        self.last_reward_piece_idx = -1
        # 下一个可用步骤
        self.availables=self.get_availables()
        # 显示mcts中间过程
        self.show_mcts_process = False

        # 盘面的状态
        # self.status = [] #deque(maxlen=10)
        # _board = np.zeros((self.height, self.width))
        # for _ in range(8):
        #     self.status.append(_board)
        self.status = np.zeros((3, self.height, self.width))
        self.set_status()
        # key
        self.set_key()   

    # 状态一共8层， 0 下一个方块， 1 是背景 ，剩下得是 6 步下落的方块
    def set_status(self):
        self.status[0]=self.get_fallpiece_board()
        self.status[1]=self.getBoard()
        self.status[2]=self.get_nextpiece_borad()

        # self.status.append(self.get_fallpiece_board())
        # del self.status[2]
        # if self.state!=0 or init:
        #     self.status[0]=self.get_nextpiece_borad()
        #     self.status[1]=self.getBoard()

        # del self.status[1]
        # if self.state!=0 or init:
        #     self.status[0]=self.get_nextpiece_borad()
            # idx = self.piececount-self.last_reward_piece_idx
            # idx = min(self.height, idx)
            # for h in range(idx):
            #     for w in range(self.width):
            #         y = self.height-h-1
            #         self.status[0][y][w] = 1 if self.status[0][y][w]==0 else 0

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
        
        return acts         

    # def getNextStatus(self):
    #     actions = self.availables
    #     fallpiece = copy.deepcopy(self.fallpiece)
    #     result={}
    #     for action in actions:
    #         if action == KEY_LEFT: fallpiece['x'] -=1
    #         if action == KEY_RIGHT: fallpiece['x'] +=1
    #         if action == KEY_DOWN: fallpiece['y'] +=1
    #         if action == KEY_ROTATION: fallpiece['rotation'] = (fallpiece['rotation'] +1)%len(pieces[self.fallpiece['shape']])
    #         if self.tetromino.validposition(self.board, fallpiece, ay=1): fallpiece['y'] +=1
    #         status = [None for i in range(8)]
    #         status[0] = self.status[0]            
    #         for i in range(1,8-1,1):
    #             status[i]=self.status[i+1]
    #         status[-1]=self.get_fallpiece_board(fallpiece)
    #         key = hash((status[-1]+status[0]).data.tobytes())
    #         result[key]=status
    #     return result

    def step(self, action, env=None):
        # 状态 0 下落过程中 1 更换方块 2 结束一局
        
        self.reward = 0
        self.steps += 1
        self.piecesteps += 1
        # self.level, self.fallfreq = self.tetromino.calculate(self.score)       

        if action == KEY_LEFT:# and self.tetromino.validposition(self.board, self.fallpiece, ax=-1):
            self.fallpiece['x'] -= 1

        if action == KEY_RIGHT:# and self.tetromino.validposition(self.board, self.fallpiece, ax=1):
            self.fallpiece['x'] += 1  

        if action == KEY_DOWN:# and self.tetromino.validposition(self.board, self.fallpiece, ay=1):
            n = 1
            if self.piecesteps>1 and self.actions[-1]==KEY_DOWN:
                while self.tetromino.validposition(self.board, self.fallpiece, ay=n+1):
                    n += 1
            self.fallpiece['y'] += n  

        if action == KEY_ROTATION:
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            # if not self.tetromino.validposition(self.board,self.fallpiece):
                # self.fallpiece['rotation'] = (self.fallpiece['rotation'] - 1) % len(pieces[self.fallpiece['shape']])
                
        self.actions.append(action)
                
        isFalling=True
        if self.tetromino.validposition(self.board, self.fallpiece, ay=1):
            self.fallpiece['y'] += 1
            if not self.tetromino.validposition(self.board, self.fallpiece, ay=1):
                isFalling = False
        else:
            isFalling = False

        # self.fallpieceheight = 20 - self.fallpiece['y']

        self.set_status()
        self.set_key()

        if not isFalling:
            self.tetromino.addtoboard(self.board, self.fallpiece)            

            self.reward = self.tetromino.removecompleteline(self.board) 
            if self.reward>0: self.last_reward_piece_idx = self.piececount         
            self.score += self.reward
            # self.pieceheight = self.getAvgHeight()  
            # self.failLines = self.getFailLines()  
            self.emptyCount = self.getSimpleEmptyCount()   
            self.heightDiff = self.getHeightDiff()
            # self.heightStd = self.getHeightStd()   
            # self.pieces_height.append(self.fallpieceheight)
            self.failtop = self.getFailTop()
            self.state = 1
            self.piecesteps = 0
            self.piececount += 1 

            self.fallpiece = self.nextpiece
            self.nextpiece = self.tetromino.getnewpiece()
            self.availables = [KEY_DOWN]
            self.actions = []
        else:
            self.state = 0

        if  env:
            env.checkforquit()
            env.render(self.board, self.score, self.level, self.fallpiece, self.nextpiece)

        if not isFalling and (not self.tetromino.validposition(self.board, self.fallpiece, ay=1) \
                            #   or (self.piececount > (self.score*2.5+self.must_reward_piece_count))  
                              ):                                      
            self.terminal = True 
            self.state = 1
            return self.state, self.reward 
        
        self.availables = self.get_availables()    

        return self.state, self.reward

    def set_key(self):
        # info = self.status[-1]+self.status[1]
        # self.key = hash(info.data.tobytes())+self.id
        self.key = hash(self.current_state().data.tobytes())

    def get_key(self):
        return self.key

    def is_status_optimal(self):
        return self.piececount<=self.score*2.5+self.must_reward_piece_count

    # 最差分 0 ~ -1
    def get_final_reward(self):
        # return -self.max_pieces_count/self.piececount+1
        return (self.piececount-self.max_pieces_count)*self.get_singe_piece_value()
    
    # 每一个方块的价值
    def get_singe_piece_value(self):
        return 1./self.max_pieces_count
    
    # 打印
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
        print("score:", self.score, "piececount:", self.piececount, "emptyCount:", self.emptyCount, "isRandomNextPiece:", self.isRandomNextPiece, \
            "must_reward_piece_count:", self.must_reward_piece_count, "exreward:", self.exreward, "failTop:",self.failtop,"heightDiff:",self.heightDiff)

    # 统计不可消除行的数量
    def getFailLines(self):
        failLines=set()
        for x in range(self.width):
            block = False
            for y in range(self.height):
                if self.board[x][y]!=blank:
                    block=True
                elif block:
                    failLines.add(y)
        return len(failLines)

    # 统计不可消除行的最高高度
    def getFailTop(self):
        blocks = [False for x in range(self.width)]
        for y in range(self.height):
            for x in range(self.width):            
                if self.board[x][y]!=blank:
                    blocks[x]=True
                elif blocks[x]==True:
                    return self.height-y
        return 0

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
                    h[x]=(self.height-y)
                    break
        # 修复夹壁>2
        for i in range(self.width):
            if i==0:
                h[i]= max(h[i], h[i+1]-2)
            elif i==self.width-1:
                h[i]= max(h[i], h[i-1]-2)
            else:
                h[i]= max(h[i], min(h[i-1]-2, h[i+1]-2))

        h_mean = np.mean(h)
        return h_mean

    # 统计高度标准差,按照碗型
    def getHeightStd(self):
        # h = np.zeros((self.width+2))
        # for x in range(self.width):
        #     for y in range(self.height):
        #         if self.board[x][y]!=blank:
        #             h[x+1]=self.height-y
        #             break
        # h[0]=h[2]
        # h[-1]=h[-3]
        # v = np.zeros((self.width))
        # for x in range(self.width):
        #     v[x] = max(abs(h[x]-h[x+1]) , abs(h[x+2]-h[x+1]))
    
        h = np.zeros((self.width))
        for x in range(self.width):            
            for y in range(self.height):
                if self.board[x][y]!=blank:
                    h[x]=self.height-y
                    break

        v = [abs(h[4]-h[5])]
        v.append((h[1]-h[0]) if h[1]-h[0]>0 else max(0,h[0]-h[1]-3))
        v.append((h[2]-h[1]) if h[2]-h[1]>0 else max(0,h[1]-h[2]-3))
        v.append((h[3]-h[2]) if h[3]-h[2]>0 else max(0,h[2]-h[3]-3))
        v.append((h[4]-h[3]) if h[4]-h[3]>0 else max(0,h[3]-h[4]-3))
        v.append((h[5]-h[6]) if h[5]-h[6]>0 else max(0,h[6]-h[5]-3))
        v.append((h[6]-h[7]) if h[6]-h[7]>0 else max(0,h[7]-h[6]-3))
        v.append((h[7]-h[8]) if h[7]-h[8]>0 else max(0,h[8]-h[7]-3))
        v.append((h[8]-h[9]) if h[8]-h[9]>0 else max(0,h[9]-h[8]-3))
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
        v.append(abs(h[1]-h[0]))
        v.append(abs(h[2]-h[1]))
        v.append(abs(h[3]-h[2]))
        v.append(abs(h[4]-h[3]))
        v.append(abs(h[5]-h[4]))
        v.append(abs(h[6]-h[5]))
        v.append(abs(h[7]-h[6]))
        v.append(abs(h[8]-h[7]))
        v.append(abs(h[9]-h[8]))
        v.remove(max(v))
        return max(v)

    def getSimpleEmptyCount(self):
        c = 0
        h = np.zeros((self.width+2))
        hs = []
        for x in range(self.width):
            l_c = -1
            for y in range(self.height):
                if self.board[x][y] == blank:
                    if l_c>=0:
                        if y not in hs: 
                            l_c += 1
                            hs.append(y)
                        else:
                            l_c += 0.1
                elif l_c==-1:
                    l_c = 0
                    h[x+1]=self.height-y
            if l_c>0: c+=l_c

        # # 加上夹壁
        # h[0]=20
        # h[-1]=20
        # for x in range(self.width):
        #     _c=min(h[x]-h[x+1],h[x+2]-h[x+1]) 
        #     if _c>=3:
        #         c += _c-2
        return c

    # 统计空洞的个数
    # 空洞最高点+空洞的最高点总数/10
    def getEmptyCount(self):
        # 每高度的的空洞数
        c = np.zeros((self.height))
        # 每列的高度
        h = np.zeros((self.width+2))
        for x in range(self.width):
            find_block=False
            for y in range(self.height):
                if find_block==False and self.board[x][y]!=blank:
                    find_block = True
                    h[x+1] = self.height-y
                if find_block and self.board[x][y]==blank: 
                    c[self.height-y] += 1

                    # if self.height-y>c_h: c_h = self.height-y
        # 加上夹壁
        h[0]=h[2]
        h[-1]=h[-3]
        for x in range(self.width):
            _c=min(h[x]-h[x+1],h[x+2]-h[x+1]) 
            if _c>=3:
                k = int(min(h[x],h[x+2])-2)
                c[k] += 1

        for x in range(self.height-1,0,-1):
            if c[x]>0:
                return x + c[x]/10       
        return 0
    
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
    def get_fallpiece_board(self, fallpiece=None):   
        board=np.zeros((self.height, self.width))
        # 需要加上当前下落方块的值
        if fallpiece==None: fallpiece = self.fallpiece

        if fallpiece != None:
            piece = fallpiece
            shapedraw = pieces[piece['shape']][piece['rotation']]
            for x in range(templatenum):
                for y in range(templatenum):
                    if shapedraw[y][x]!=blank:
                        px, py = x+piece['x'], y+piece['y']
                        if px>=0 and py>=0:
                            board[py][px]=1
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
        # return self.status
        return np.array(self.status)
