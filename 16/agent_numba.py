import hashlib
import numpy as np
import random
import time
# from numba import njit

boxsize = 20
boardwidth = 10
boardheight = 20
templatenum = 5
blank = 0

stemplate = np.array([
            [[0,0,0,0,0],
             [0,0,1,1,0],
             [0,1,1,0,0],
             [0,0,0,0,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,0,1,0,0],
             [0,0,1,1,0],
             [0,0,0,1,0],
             [0,0,0,0,0]]
            ],dtype=np.int8)
 
ztemplate = np.array([
            [[0,0,0,0,0],
             [0,1,1,0,0],
             [0,0,1,1,0],
             [0,0,0,0,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,0,0,1,0],
             [0,0,1,1,0],
             [0,0,1,0,0],
             [0,0,0,0,0]]
            ],dtype=np.int8)
 
itemplate = np.array([
            [[0,0,1,0,0],
             [0,0,1,0,0],
             [0,0,1,0,0],
             [0,0,1,0,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,1,1,1,1],
             [0,0,0,0,0],
             [0,0,0,0,0],
             [0,0,0,0,0]]
            ],dtype=np.int8)
 
otemplate = np.array([
            [[0,0,0,0,0],
             [0,0,1,1,0],
             [0,0,1,1,0],
             [0,0,0,0,0],
             [0,0,0,0,0]]
            ],dtype=np.int8)
 
ltemplate = np.array([
            [[0,0,0,0,0],
             [0,0,1,0,0],
             [0,0,1,0,0],
             [0,0,1,1,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,0,0,1,0],
             [0,1,1,1,0],
             [0,0,0,0,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,0,1,1,0],
             [0,0,0,1,0],
             [0,0,0,1,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,1,1,1,0],
             [0,1,0,0,0],
             [0,0,0,0,0],
             [0,0,0,0,0]]
            ],dtype=np.int8)
 
jtemplate = np.array([
            [[0,0,0,0,0],
             [0,0,1,0,0],
             [0,0,1,0,0],
             [0,1,1,0,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,1,1,1,0],
             [0,0,0,1,0],
             [0,0,0,0,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,0,1,1,0],
             [0,0,1,0,0],
             [0,0,1,0,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,1,0,0,0],
             [0,1,1,1,0],
             [0,0,0,0,0],
             [0,0,0,0,0]]
            ],dtype=np.int8)
 
ttemplate = np.array([
            [[0,0,0,0,0],
             [0,0,1,0,0],
             [0,1,1,1,0],
             [0,0,0,0,0],
             [0,0,0,0,0]],
            [[0,0,1,0,0],
             [0,1,1,0,0],
             [0,0,1,0,0],
             [0,0,0,0,0],
             [0,0,0,0,0]],
            [[0,0,0,0,0],
             [0,1,1,1,0],
             [0,0,1,0,0],
             [0,0,0,0,0],
             [0,0,0,0,0]],
            [[0,0,1,0,0],
             [0,0,1,1,0],
             [0,0,1,0,0],
             [0,0,0,0,0],
             [0,0,0,0,0]]
            ],dtype=np.int8)

pieces = {'s':stemplate,
          'z':ztemplate,
          'i':itemplate,
          'o':otemplate,
          'l':ltemplate,
          'j':jtemplate,
          't':ttemplate}

pieces_type = list(pieces.keys())

KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_NONE, KEY_DOWN = 0, 1, 2, 3, 4
ACTIONS = [KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_NONE, KEY_DOWN]
ACTIONS_NAME = ["O","L","R","N","D"]
ACTONS_LEN = len(ACTIONS)

# @njit(cache=True)
def nb_calc_down_count(board, piece, piece_x, piece_y, templatenum=5):
    p_h= np.sum(piece,axis=0, dtype=np.int8) + np.argmax(piece,axis=0)
    _board = np.copy(board)
    if piece_y>0:
        _board[:piece_y]=0
    b = np.argmax(_board, axis=0)
    b[b==0]=20
    c = np.zeros_like(b, dtype=np.int8)    
    for x in range(templatenum):
        if p_h[x]>0:
            c[piece_x+x] = p_h[x]
    b[c==0]=20
    min_y=np.min(b-c)
    count = min_y-piece_y 
    return count

# @njit()
def nb_validposition(board, piece, piece_x, piece_y, ax=0, ay=0, templatenum=5, boardwidth=10, boardheight=20):
    oneboard = np.ones((boardheight+6, boardwidth+12), dtype=np.int8)
    oneboard[0:-6,6:-6]=0
    oneboard[0:-6,6:-6] += board   
    for y in range(templatenum):
        for x in range(templatenum):
            oneboard[y+piece_y+ay][6+x+piece_x+ax] += piece[y][x]
    return not np.any(oneboard[0:-5,5:-5] > 1)
    
    # for y in range(templatenum-1,-1,-1):
    #     for x in range(templatenum):
    #         if piece[y][x]!=0: 
    #             _x = x + piece_x + ax
    #             _y = y + piece_y + ay
    #             if _y<0: continue
    #             if _x<0 or _x>=boardwidth or _y>=boardheight or board[_y][_x]!=0:
    #                 return False
    # return True

# @njit()
def nb_addtoboard(board,piece,piece_x,piece_y,boardwidth=10,boardheight=20):
    for x in range(templatenum):
        for y in range(templatenum):
            if piece[y][x]!=0:
                w = x + piece_x
                h = y + piece_y
                if w>=0 and w<boardwidth and h>=0 and h<boardheight:
                    board[h][w] = 1    

# @njit(cache=True)
def nb_removecompleteline(board, boardheight=20):
    numremove = 0
    y = boardheight-1
    while y >=0:
        if np.min(board[y])==1:
            for pulldowny in range(y, 0, -1):
                board[pulldowny] = board[pulldowny-1]
            board[0] = 0
            numremove+=1
        else:
            y-=1
    return numremove

# @njit(cache=True)
def nb_get_status(piece, p_x, p_y, templatenum=5):
    status=np.zeros((boardheight, boardwidth), dtype=np.int8)
    for x in range(templatenum):
        for y in range(templatenum):
            if piece[y][x]==1:
                px, py = x+p_x, y+p_y
                if px>=0 and px<10 and py>=0 and py<20:
                    status[py][px]=1
    return status                        

# @njit(cache=True)
def nb_put_status(board,piece, p_x, p_y, templatenum=5):
    for x in range(templatenum):
        for y in range(templatenum):
            if piece[y][x]==1:
                px, py = x+p_x, y+p_y
                if px>=0 and px<10 and py>=0 and py<20:
                    board[py][px]=1

# 统计为空的个数 统计最高的空的高度
# @njit(cache=True)
def nb_getEmptyCount(board):
    # blank_line = np.zeros((boardwidth), dtype=np.int8)
    # values = np.sum(board,axis=1)
    # for y in range(boardheight):
    #     if values[y] == 0: continue
    #     for x in range(boardwidth):
    #         if board[y][x] > 0:
    #             blank_line[x] = 1
    #         elif blank_line[x] == 1:
    #             return 20-y
    # return 0
    d=(20-np.argmax(board,axis=0)-np.sum(board,axis=0))
    return np.sum(d[d!=20])

# def nb_getEmptyCount2(board):
#     v = 0 
#     for i in range(boardwidth):
#         s = 0
#         for j in range(boardheight):
#             if board[j][i] == 1:
#                 s = 1
#             else:
#                 v += s*1.01**(19-j)
#     return v

# 统计占位的个数
# @njit(cache=True)
def nb_getUsedCount(board):
    d=np.argmax(board,axis=0)
    d[d==0]=20
    d = 20 - d
    
    # s = 0
    # c = d[1]-d[0]
    # if c>3:
    #     s += c-3
    # c = d[8]-d[9]
    # if c>3:
    #     s += c-3
    # for i in range(8):    
    #     c = min(d[i],d[i+2])-d[i+1]
    #     if c>3:
    #         s += c-3        
    # return np.sum(d)+s
    return np.sum(d)
                
# 统计最高位置以下的所有空的方块
def nb_getTerminalEmptyCount(board):
    d = np.argmax(board,axis=0)
    max_h = 20-np.min(d[d!=0])
    d = max_h*10-np.sum(board)
    return d

# 统计不可消除行的数量
# @njit(cache=True)
def nb_getFailLines(board):
    failLines=set()
    for x in range(boardwidth):
        block = False
        for y in range(boardheight):
            if board[y][x]!=blank:
                block=True
            elif block:
                failLines.add(y)
    return len(failLines)

# 统计不可消除行的最高高度
# @njit(cache=True)
def nb_getFailTop(board):
    blocks = [False for x in range(boardwidth)]
    for y in range(boardheight):
        for x in range(boardwidth):            
            if board[y][x]!=blank:
                blocks[x]=True
            elif blocks[x]==True:
                return boardheight-y
    return 0

# 统计当前最大高度
# @njit(cache=True)
def nb_getMaxHeight(board):
    c = -1
    for y in range(boardheight):
        for x in range(boardwidth):
            if board[y][x]!=blank:
                c=y
                break
        if c!=-1:break  
    h = 0 if c == -1 else boardheight - c                          
    return h

# 统计当前平均高度
# @njit(cache=True)
def nb_getAvgHeight(board, std=False):
    h = np.zeros((boardwidth), dtype=np.int8)
    for x in range(boardwidth):
        for y in range(boardheight):
            if board[y][x]!=blank:
                h[x]=(boardheight-y)
                break
    # 修复夹壁>2
    for i in range(boardwidth):
        if i==0:
            h[i]= max(h[i], h[i+1]-2)
        elif i==boardwidth-1:
            h[i]= max(h[i], h[i-1]-2)
        else:
            h[i]= max(h[i], min(h[i-1]-2, h[i+1]-2))

    h_mean = np.mean(h)
    return h_mean

# 统计高度标准差,按照碗型
# @njit(cache=True)
def nb_getHeightStd(board):
    h = np.zeros((boardwidth), dtype=np.int8)
    for x in range(boardwidth):            
        for y in range(boardheight):
            if board[y][x]!=blank:
                h[x]=boardheight-y
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
# @njit(cache=True)
def nb_getHeightDiff(board):
    h = np.zeros((boardwidth), dtype=np.int8)
    for x in range(boardwidth):            
        for y in range(boardheight):
            if board[y][x]!=blank:
                h[x]=boardheight-y
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

# @njit(cache=True)
def nb_getSimpleEmptyCount(board):
    c = 0
    h = np.zeros((boardwidth+2), dtype=np.int8)
    hs = []
    for x in range(boardwidth):
        l_c = -1
        for y in range(boardheight):
            if board[y][x] == blank:
                if l_c>=0:
                    if y not in hs: 
                        l_c += 1
                        hs.append(y)
                    else:
                        l_c += 0.1
            elif l_c==-1:
                l_c = 0
                h[x+1]=boardheight-y
        if l_c>0: c+=l_c
    return c

# 统计空洞的个数
# 空洞最高点+空洞的最高点总数/10
# @njit(cache=True)
def nb_getEmptyCount2(board):
    # 每高度的的空洞数
    c = np.zeros((boardheight), dtype=np.int8)
    # 每列的高度
    h = np.zeros((boardwidth+2), dtype=np.int8)
    for x in range(boardwidth):
        find_block=False
        for y in range(boardheight):
            if find_block==False and board[y][x]!=blank:
                find_block = True
                h[x+1] = boardheight-y
            if find_block and board[y][x]==blank: 
                c[boardheight-y] += 1

                # if self.height-y>c_h: c_h = self.height-y
    # 加上夹壁
    h[0]=h[2]
    h[-1]=h[-3]
    for x in range(boardwidth):
        _c=min(h[x]-h[x+1],h[x+2]-h[x+1]) 
        if _c>=3:
            k = int(min(h[x],h[x+2])-2)
            c[k] += 1

    for x in range(boardheight-1,0,-1):
        if c[x]>0:
            return x + c[x]/10       
    return 0

class Agent():
    def __init__(self, isRandomNextPiece=False, nextPiecesList=[]):
        self.nextPieceList=[p for p in nextPiecesList]
        self.next_Pieces_list_len=len(nextPiecesList)
        self.isRandomNextPiece=isRandomNextPiece
        self.pieceCount = 0
        self.piecehis=[]
        # 下落的方块
        self.fallpiece = self.getnewpiece()
        # 下一个待下落的方块
        self.nextpiece = self.getnewpiece()
        # 是否结束
        self.terminal = False
        # 得分
        self.score = 0
        # 消行数
        self.removedlines = 0
        # 全部步长
        self.steps = 0
        # 每个方块的步长
        self.piecesteps = 0
        # 方块的数量
        self.piececount = 0
        # 方块的最高高度
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
        # 限制步骤
        self.limitstep=False
        # 面板
        self.board = self.getblankboard()
        # 状态： 0 下落过程中 1 更换方块 2 结束一局
        self.state = 0
        self.piece_actions = "" 
        # 当前prices所有动作
        # self.actions=[]
        self.cache=None

        # 显示mcts中间过程
        self.show_mcts_process = False
        # 记录下降的次数
        self.downcount=0
        # 继续上一个方块的状态
        self.last_reward = -1
        # 盘面的状态
        self.need_update_status=True
        self.status = np.zeros((4, boardheight, boardwidth), dtype=np.int8)
        self.set_status()
        # key
        self.key = 0
        self.full_key = 0
        self.set_key()
        # 下一个可用步骤，设置需要在key之后
        self.availables=np.ones(ACTONS_LEN, dtype=np.int8)
        self.set_availables()
        self.start_time = time.time()

    def clone(self):
        agent = Agent()
        agent.nextPieceList=[p for p in self.nextPieceList]
        agent.next_Pieces_list_len = self.next_Pieces_list_len
        agent.isRandomNextPiece = self.isRandomNextPiece
        agent.pieceCount = self.pieceCount 
        agent.piecehis = [p for p in self.piecehis]
        agent.fallpiece = self.clonePiece(self.fallpiece)
        agent.nextpiece = self.clonePiece(self.nextpiece)
        agent.terminal = self.terminal
        agent.score = self.score
        agent.removedlines = self.removedlines
        agent.steps = self.steps
        agent.piecesteps = self.piecesteps
        agent.piececount = self.piececount
        agent.pieceheight = self.pieceheight
        agent.emptyCount = self.emptyCount
        agent.heightDiff = self.heightDiff
        agent.heightStd = self.heightStd
        agent.failLines = self.failLines
        agent.failtop = self.failtop
        agent.fallpieceheight = self.fallpieceheight
        agent.limitstep=self.limitstep
        agent.board=np.copy(self.board)
        agent.state=self.state
        agent.piece_actions = self.piece_actions
        agent.availables=np.copy(self.availables)
        agent.show_mcts_process = self.show_mcts_process
        agent.downcount = self.downcount
        agent.last_reward = self.last_reward
        agent.need_update_status = self.need_update_status
        agent.status = np.copy(self.status)
        agent.key = self.key
        agent.full_key = self.full_key        
        agent.cache = self.cache
        agent.start_time = self.start_time
        return agent
    
    def clonePiece(self, piece):
        if piece==None: return None
        p={}
        p["shape"]=piece["shape"]
        p["rotation"]=piece["rotation"]
        p["x"]=piece["x"]
        p["y"]=piece["y"]
        return p
        
    def getpiece(self, shape=None):
        if shape==None:
            shape = random.choice(list(pieces.keys()))
        newpiece = {
                    'shape':shape,
                    'rotation': 0,
                    'x': int(boardwidth)//2-int(templatenum//2),
                    'y': -1,
                   }
        return newpiece
        
    def getnewpiece(self):
        if not self.isRandomNextPiece:
            if len(self.nextPieceList)<100:
                for _ in range(99):
                    self.nextPieceList.append(random.choice(list(pieces.keys()))) 
        if len(self.nextPieceList)>0:
            nextpieceshape = self.nextPieceList.pop(0)
            nextpiece = self.getpiece(nextpieceshape)
        else:
            nextpiece = self.getpiece()  
        self.piecehis.append(nextpiece["shape"])
        self.pieceCount += 1
        return nextpiece
    
    def getblankboard(self):
        board = np.zeros((boardheight,boardwidth),dtype=np.int8)
        return board
    
    def addtoboard(self,board,piece):
        _piece = pieces[piece['shape']][piece['rotation']]
        nb_addtoboard(board, _piece, piece['x'], piece['y'])               
        
    def validposition(self, board, piece, ax = 0, ay = 0):        
        _piece = pieces[piece['shape']][piece['rotation']]
        return nb_validposition(board, _piece, piece['x'], piece['y'], ax=ax, ay=ay)
    
    def removecompleteline(self, board):
        return nb_removecompleteline(board)
    
    def calc_down_count(self,board,piece):
        _piece = pieces[piece['shape']][piece['rotation']]
        return nb_calc_down_count(board, _piece, piece['x'], piece['y'])

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
    def set_availables(self):
        if self.cache!=None and self.key in self.cache:
            c = self.cache[self.key]
            self.availables = np.copy(c)
            return
        
        # acts=[KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_NONE, KEY_DOWN]
        if not self.validposition(self.board, self.fallpiece, ax = -1):
            self.availables[KEY_LEFT]=0
        else:
            self.availables[KEY_LEFT]=1
            # acts.remove(KEY_LEFT)
        if not self.validposition(self.board, self.fallpiece, ax = 1):
            self.availables[KEY_RIGHT]=0
        else:
            self.availables[KEY_RIGHT]=1
            # acts.remove(KEY_RIGHT)   
        # if not self.validposition(self.board, self.fallpiece, ay = 1):
        #     acts.remove(KEY_DOWN)

        if self.fallpiece['shape']=="o":
            self.availables[KEY_ROTATION]=0
            # acts.remove(KEY_ROTATION)
        else:            
            r = self.fallpiece['rotation']
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.validposition(self.board,self.fallpiece):
                self.availables[KEY_ROTATION]=0
            else:
                self.availables[KEY_ROTATION]=1
                # acts.remove(KEY_ROTATION)
            self.fallpiece['rotation'] = r

        # if not KEY_DOWN in acts : acts.append(KEY_DOWN)

        if self.cache!=None:
            self.cache[self.key]=np.copy(self.availables)

    # 设置缓存
    def setCache(self, cache):
        self.cache=cache
        
    def step(self, action):                
        # 状态 0 下落过程中 1 更换方块 2 结束一局    
        self.steps += 1
        self.piecesteps += 1
        # self.level, self.fallfreq = self.calculate(self.score)
        
        # self.actions.append(action)
        if self.availables[action] == 1:
            if action == KEY_LEFT: 
                self.fallpiece['x'] -= 1

            if action == KEY_RIGHT:
                self.fallpiece['x'] += 1  

            if action == KEY_ROTATION:
                self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])

            if action == KEY_DOWN:
                while True: 
                    self.downcount += 1
                    self.fallpiece['y'] += 1            
                    if not self.validposition(self.board, self.fallpiece, ay=1): break

        isFalling=True
        if self.validposition(self.board, self.fallpiece, ay=1):
            self.fallpiece['y'] += 1
            if not self.validposition(self.board, self.fallpiece, ay=1):
                isFalling = False
        else:
            isFalling = False

        # self.fallpieceheight = 20 - self.fallpiece['y']
        if self.piecesteps==1: self.piece_actions=""
        self.piece_actions += self.position_to_action_name(action)

        reward = 0
        removedlines = 0
        if not isFalling:   
            self.addtoboard(self.board, self.fallpiece)            
            removedlines = self.removecompleteline(self.board)
            emptyCount = self.getEmptyCount()               
                    
            self.need_update_status=True
            # if removedlines>0: print("OK!!!",removedlines)
            self.removedlines += removedlines
            reward = removedlines            
            
            # 鼓励垂直下落和连续多次消行和消除空格
            # if removedlines>0 and not putEmptyBlock:
            if removedlines>0:
                # 如果消行了，奖励加上从上一个阶段起总共下降的个数
                reward += self.downcount*0.01
                
                # 如果消除空的气泡了
                # if emptyCount<self.emptyCount:
                #     reward += self.emptyCount-emptyCount
                    
                # 如果上一方块也消行了，奖励加1
                if self.piececount-self.last_reward==1: reward += 1 
                
                self.downcount = 0
                self.last_reward = self.piececount
                
            # reward -= (emptyCount - self.emptyCount)*0.5  
            self.emptyCount  = emptyCount
            
            # self.score += reward    # 一个方块1点 
            # self.score = -nb_getUsedCount(self.board)/4 #+ self.piececount + 1  # 一个方块1点 

            self.pieceheight = self.getMaxHeight()  
            # self.failLines = self.getFailLines()  
            # self.heightDiff = self.getHeightDiff()
            # self.heightStd = self.getHeightStd()   
            # self.failtop = self.getFailTop()
            self.state = 1
            self.piecesteps = 0
            self.piececount += 1 

            # _fallpiece_shape = self.fallpiece['shape']
            self.fallpiece = self.nextpiece
            self.nextpiece = self.getnewpiece()            
            # self.actions = []

            # 这里强制20个方块内必需合成一行个 和 强制第5个方块之后不能有空
            if not self.validposition(self.board, self.fallpiece, ay=0):# or (self.limitstep and self.emptyCount>10):
                self.terminal = True 
                self.state = 1
                removedlines = -1
            # if (self.piececount-self.last_reward>=20) or \
            #     (putEmptyBlock and reward==0 and self.piececount>5) or \
            #     (putEmptyBlock and reward==0 and self.piececount<=5 and self.emptyCount>2) or \
            #     (putEmptyBlock and _fallpiece_shape in ["l","j","i","o","t"]):
            #         removedlines = -1
            #         if self.limitstep:
            #             self.terminal = True 
            #             self.state = 1
        else:
            self.state = 0

        # 以下顺序不能变
        self.set_status()
        self.set_key()       
        self.set_availables()

        return self.state, removedlines

    # 状态一共4层， 0 当前下落方块， 1 是背景 ，2下落方块的上一个动作. 3是下一个方块
    def set_status(self):
        # status = np.zeros((4, boardheight, boardwidth), dtype=np.int8)        
        piece = self.fallpiece
        shapedraw = pieces[piece['shape']][piece['rotation']]
        self.status[2] = self.status[0] | self.status[1]
        self.status[0] = nb_get_status(shapedraw, piece['x'], piece['y'])        

        # 更新历史动作，仅仅限于同一个方块周期，如果是第二步更新上一步的背景
        # if self.piecesteps == 1:
        #     self.status[2] = self.board.copy()            
        # self.status[2] = self.status[2] | self.status[0]
        
        # 如果方块落地了,更新背景方块和下一个方块 
        if self.piecesteps == 0:
            self.status[1] = self.board.copy()
            self.status[3] = np.zeros((boardheight, boardwidth), dtype=np.int8)  
            piece_template = pieces[self.nextpiece['shape']]
            for i in range(4):
                _rotation = i%len(piece_template)
                _piece = piece_template[_rotation]
                nb_put_status(self.status[3], _piece, 0, i*5)
            piece_template = pieces[self.fallpiece['shape']]
            for i in range(4):
                _rotation = i%len(piece_template)
                _piece = piece_template[_rotation]
                nb_put_status(self.status[3], _piece, 5, i*5)               

    def set_key(self):
        # board = self.status[0] | self.status[1] # np.sum(self.status[:2],axis=0)        
        # key = np.packbits(board).tobytes()
        # key = int.from_bytes(key, byteorder='big')  # 转换为整数
        
        # keylist=[]
        # for b in board.flat:
        #     keylist.append(str(b))
        # key = int("".join(keylist), 2)
        
        # board = self.status[1]
        # key = np.packbits(board).tobytes()
        # key = int.from_bytes(key, byteorder='big')
        # key = key*10 + pieces_type.index(self.fallpiece['shape'])
        # key = key*10 + self.fallpiece['rotation']
        # key = key*10 + self.fallpiece['x']
        # key = key*20 + self.fallpiece['y']
        # key = key*10 + pieces_type.index(self.nextpiece['shape'])
        # self.key = key
                
        bytes = self.status[[0,1,3]].tobytes()
        key = hashlib.md5(bytes).hexdigest()
        key = int(key, 16)
        self.key = key
        
        bytes = self.status.tobytes()
        key = hashlib.md5(bytes).hexdigest()
        key = int(key, 16)
        self.full_key = key
        
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
        # print("actions:" ,self.piece_actions)
        print("")
        board = np.copy(self.board)
        for x in range(templatenum):
            for y in range(templatenum):
                w = x + self.fallpiece['x']
                h = y + self.fallpiece['y']
                if pieces[self.fallpiece['shape']][self.fallpiece['rotation']][y][x]!=blank:
                    if w>=0 and w<boardwidth and h>=0 and h<boardheight:
                        board[h][w] = 1

        for y in range(boardheight):
            line="| "
            for x in range(boardwidth):
                if board[y][x]==blank:
                    line=line+"  "
                else:
                    line=line+str(board[y][x])+" "
            print(line)
        print(" "+" -"*boardwidth)
        print("pieceheight:", self.pieceheight, "lines:",self.removedlines, "piececount:", self.piececount, "piecehis:", len(self.piecehis), "/", \
            self.next_Pieces_list_len, "steps:", self.steps, "rewardlimit:", self.piececount-self.last_reward)


    def getTerminalEmptyCount(self):    
        return nb_getTerminalEmptyCount(self.board)

    # 统计不可消除行的数量
    def getFailLines(self):
        return nb_getFailLines(self.board)

    # 统计不可消除行的最高高度
    def getFailTop(self):
        return nb_getFailTop(self.board)

    # 统计当前最大高度
    def getMaxHeight(self):
        return nb_getMaxHeight(self.board)

    # 统计当前平均高度
    def getAvgHeight(self, std=False):
        return nb_getAvgHeight(self.board, std)

    # 统计高度标准差,按照碗型
    def getHeightStd(self):
        return nb_getHeightStd(self.board)   

    # 统计数据相邻差值
    def getHeightDiff(self):
        return nb_getHeightDiff(self.board)

    def getSimpleEmptyCount(self):
        return nb_getSimpleEmptyCount(self.board)

    # 统计空洞的个数
    # 空洞最高点+空洞的最高点总数/10
    def getEmptyCount(self):
        return nb_getEmptyCount(self.board)
    
    # 获得当前局面信息
    def getBoard(self):
        return self.board

    # 获得下落方块的信息
    def get_fallpiece_board(self, fallpiece=None):   
        board=self.getblankboard()
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
        return board

    # # 获得待下落方块的信息
    def get_nextpiece_borad(self):
        board=self.getblankboard()
        off = int(boardwidth)//2-int(templatenum//2)
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
        return self.status

