import numpy as np
import random
from numba import njit

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

class Tetromino():  
    def __init__(self, isRandomNextPiece=True, nextPieceList=[]):
        self.nextPieceList=nextPieceList
        self.isRandomNextPiece=isRandomNextPiece
        self.pieceCount = 0
        self.piecehis=[]

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
        for x in range(templatenum):
            for y in range(templatenum):
                w = x + piece['x']
                h = y + piece['y']
                if pieces[piece['shape']][piece['rotation']][y][x]!=blank:
                    if w>=0 and w<boardwidth and h>=0 and h<boardheight:
                        board[h][w] = 1
                
    def onboard(self,x,y):
        return x >=0 and x<boardwidth and y<boardheight
        
    def validposition(self,board,piece,ax = 0,ay = 0):
        for x in range(templatenum):
            for y in range(templatenum):
                aboveboard = y +piece['y'] +ay < 0
                if aboveboard or pieces[piece['shape']][piece['rotation']][y][x]==blank:
                    continue
                if not self.onboard(x + piece['x']+ax,y+piece['y']+ay):
                    return False
                if board[y+piece['y']+ay][x+piece['x']+ax]!=blank:
                    return False
        return True
    
    def completeline(self,board,y):
        return np.min(board[y])==1
    
    def removecompleteline(self,board):
        numremove = 0
        y = boardheight-1
        while y >=0:
            if self.completeline(board,y):
                for pulldowny in range(y,0,-1):
                    board[pulldowny] = board[pulldowny-1]
                board[0] = blank
                numremove+=1
            else:
                y-=1
        return numremove

KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN = 0, 1, 2, 3
ACTIONS = [KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
ACTIONS_NAME = ["O","L","R","D"]
ACTONS_LEN = len(ACTIONS)

class Agent(Tetromino):
    def __init__(self, isRandomNextPiece=False, nextPiecesList=[]):
        super(Agent, self).__init__(isRandomNextPiece, nextPiecesList)  
        # 下落的方块
        self.fallpiece = self.getnewpiece()
        # 下一个待下落的方块
        self.nextpiece = self.getnewpiece()
        # 是否结束
        self.terminal = False
        # 得分
        self.score = 0
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
        self.board = self.getblankboard()
        # 状态： 0 下落过程中 1 更换方块 2 结束一局
        self.state = 0
        # 每个方块的高度
        self.pieces_height = []     
        # 当前prices所有动作
        self.actions=[]
        # 下一个可用步骤
        self.availables=self.get_availables()
        # 显示mcts中间过程
        self.show_mcts_process = False
        # 盘面的状态
        self.need_update_status=True
        self.status = np.zeros((3, boardheight, boardwidth))
        self.set_status()
        # key
        self.set_key()   
        

    # 状态一共3层， 0 下一个方块， 1 是背景 ，剩下得是 6 步下落的方块
    def set_status(self):
        self.status[0]=0
        if self.fallpiece != None:
            piece = self.fallpiece
            shapedraw = pieces[piece['shape']][piece['rotation']]
            self.status[0] = nb_get_status(shapedraw, piece['x'], piece['y'], 0)
                            
        if self.need_update_status==True:
            self.status[1]=self.board
            self.status[2]=0
            if self.nextpiece != None:
                off = int(boardwidth)//2-int(templatenum//2)
                piece = self.nextpiece  
                shapedraw = pieces[piece['shape']][piece['rotation']]
                self.status[0] = nb_get_status(shapedraw, piece['x'], piece['y'], off)
                            
            self.need_update_status=False
        
        # self.status[0]=self.get_fallpiece_board()
        # self.status[2]=self.get_nextpiece_borad()
        # self.status[1]=self.board

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

        if not self.validposition(self.board, self.fallpiece, ax = -1):
            acts.remove(KEY_LEFT)
        if not self.validposition(self.board, self.fallpiece, ax = 1):
            acts.remove(KEY_RIGHT)   
        if not self.validposition(self.board, self.fallpiece, ay = 1):
            acts.remove(KEY_DOWN)

        if self.fallpiece['shape']=="o":
            acts.remove(KEY_ROTATION)
        else:
            r = self.fallpiece['rotation']
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.validposition(self.board,self.fallpiece):
                acts.remove(KEY_ROTATION)
            self.fallpiece['rotation'] = r

        if not KEY_DOWN in acts : acts.append(KEY_DOWN)
        
        return acts         

    def step(self, action):
        # 状态 0 下落过程中 1 更换方块 2 结束一局
        
        self.steps += 1
        self.piecesteps += 1
        # self.level, self.fallfreq = self.calculate(self.score)
        
        self.actions.append(action)

        if action == KEY_LEFT:# and self.validposition(self.board, self.fallpiece, ax=-1):
            self.fallpiece['x'] -= 1

        if action == KEY_RIGHT:# and self.validposition(self.board, self.fallpiece, ax=1):
            self.fallpiece['x'] += 1  

        if action == KEY_DOWN:# and self.validposition(self.board, self.fallpiece, ay=1):
            n = 1
            while self.validposition(self.board, self.fallpiece, ay=n+1):
                n += 1
            self.fallpiece['y'] += n  
            down_count = n

        if action == KEY_ROTATION:
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            # if not self.tetromino.validposition(self.board,self.fallpiece):
                # self.fallpiece['rotation'] = (self.fallpiece['rotation'] - 1) % len(pieces[self.fallpiece['shape']])
        isFalling=True
        if self.validposition(self.board, self.fallpiece, ay=1):
            self.fallpiece['y'] += 1
            if not self.validposition(self.board, self.fallpiece, ay=1):
                isFalling = False
        else:
            isFalling = False

        # self.fallpieceheight = 20 - self.fallpiece['y']

        self.set_status()
        self.set_key()
        reward = 0
        self.score -= 0.001
        if not isFalling:            
            self.addtoboard(self.board, self.fallpiece)            
            self.need_update_status=True
            lines = self.removecompleteline(self.board)
            reward = lines * 2.5
            
            # 鼓励垂直下落
            if lines>0 and action == KEY_DOWN:
                reward += down_count*0.1    
                
            self.score += reward    # 一个方块1点 
            # self.pieceheight = self.getAvgHeight()  
            # self.failLines = self.getFailLines()  
            # self.emptyCount = self.getSimpleEmptyCount()   
            # self.heightDiff = self.getHeightDiff()
            # self.heightStd = self.getHeightStd()   
            # self.pieces_height.append(self.fallpieceheight)
            # self.failtop = self.getFailTop()
            self.state = 1
            self.piecesteps = 0
            self.piececount += 1 

            self.fallpiece = self.nextpiece
            self.nextpiece = self.getnewpiece()
            self.availables = [KEY_DOWN]
            self.actions = []
        else:
            self.state = 0

        if not isFalling and (not self.validposition(self.board, self.fallpiece, ay=1) \
                            #   or (self.piececount > (self.score*2.5+self.must_reward_piece_count))  
                              ):                                      
            self.terminal = True 
            self.state = 1
            return self.state, reward 
        
        self.availables = self.get_availables()
        return self.state, reward

    def set_key(self):
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
        print("score:", self.score, "piececount:", self.piececount, "emptyCount:", self.emptyCount, "isRandomNextPiece:", self.isRandomNextPiece, \
             "exreward:", self.exreward, "failTop:",self.failtop,"heightDiff:",self.heightDiff)

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

@njit
def nb_get_status(piece, p_x, p_y, xoff=0):
    status=np.zeros((boardheight, boardwidth))
    for x in range(templatenum):
        for y in range(templatenum):
            if piece[y][x]!=blank:
                px, py = x+p_x, y+p_y
                if px>=0 and py>=0:
                    status[py][px+xoff]=1
    return status                        

# 统计不可消除行的数量
@njit
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
@njit
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
@njit
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
@njit
def nb_getAvgHeight(board, std=False):
    h = np.zeros((boardwidth))
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
@njit
def nb_getHeightStd(board):
    h = np.zeros((boardwidth))
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
@njit
def nb_getHeightDiff(board):
    h = np.zeros((boardwidth))
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

@njit
def nb_getSimpleEmptyCount(board):
    c = 0
    h = np.zeros((boardwidth+2))
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
@njit
def nb_getEmptyCount(board):
    # 每高度的的空洞数
    c = np.zeros((boardheight))
    # 每列的高度
    h = np.zeros((boardwidth+2))
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