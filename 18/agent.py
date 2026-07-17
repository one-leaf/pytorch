import numpy as np
import random
import time

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

# piece type → color ID (1~7) for display
PIECE_IDS = {'s':1, 'z':2, 'i':3, 'o':4, 'l':5, 'j':6, 't':7}

KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_NONE, KEY_DOWN = 0, 1, 2, 3, 4
ACTIONS = [KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_NONE, KEY_DOWN]
ACTONS_LEN = len(ACTIONS)

def nb_validposition(board, piece, piece_x, piece_y, ax=0, ay=0, templatenum=5, boardwidth=10, boardheight=20):
    oneboard = np.ones((boardheight+6, boardwidth+12), dtype=np.int8)
    oneboard[0:-6,6:-6]=0
    oneboard[0:-6,6:-6] += board
    for y in range(templatenum):
        for x in range(templatenum):
            oneboard[y+piece_y+ay][6+x+piece_x+ax] += piece[y][x]
    return not np.any(oneboard[0:-5,5:-5] > 1)


class Agent():
    def __init__(self, isRandomNextPiece=False, nextPiecesList=[]):
        self.nextPieceList=[p for p in nextPiecesList]
        self.next_Pieces_list_len=len(nextPiecesList)
        self.isRandomNextPiece=isRandomNextPiece
        self.piecehis=[]
        # 下落的方块
        self.fallpiece = self.getnewpiece()
        # 下一个待下落的方块
        self.nextpiece = self.getnewpiece()
        # 是否结束
        self.terminal = False
        # 消行数
        self.removedlines = 0
        # 全部步长
        self.steps = 0
        # 每个方块的步长
        self.piecesteps = 0
        # 方块的数量
        self.piececount = 0
        # 面板
        self.board = self.getblankboard()
        # 面板碰撞掩码 (0/1)
        self.board_mask = self.getblankboard()
        # 记录下降的次数
        self.downcount=0
        # 继续上一个方块的状态
        self.last_reward = -1
        # 盘面的状态
        self.state = np.zeros((2, boardheight, boardwidth), dtype=np.int8)
        self.set_state()
        # 下一个可用步骤
        self.availables=np.ones(ACTONS_LEN, dtype=np.int8)
        self.set_availables()
        self.start_time = time.time()

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
        return nextpiece
    
    def getblankboard(self):
        board = np.zeros((boardheight,boardwidth),dtype=np.int8)
        return board
    
    def add_piece(self, piece):
        """Add piece to both colored board (1~7) and binary mask."""
        _piece = pieces[piece['shape']][piece['rotation']]
        value = PIECE_IDS[piece['shape']]
        for x in range(templatenum):
            for y in range(templatenum):
                if _piece[y][x] != 0:
                    w, h = x + piece['x'], y + piece['y']
                    if 0 <= w < boardwidth and 0 <= h < boardheight:
                        self.board[h][w] = value
                        self.board_mask[h][w] = 1

    def validposition_mask(self, piece, ax=0, ay=0):
        """Check collision using binary mask (needed when board stores 1~7)."""
        _piece = pieces[piece['shape']][piece['rotation']]
        return nb_validposition(self.board_mask, _piece, piece['x'], piece['y'], ax=ax, ay=ay)

    def remove_mask_line(self):
        """Clear completed lines from both board_mask and board."""
        numremove = 0
        y = boardheight - 1
        while y >= 0:
            if np.min(self.board_mask[y]) == 1:
                for pulldowny in range(y, 0, -1):
                    self.board_mask[pulldowny] = self.board_mask[pulldowny - 1]
                    self.board[pulldowny] = self.board[pulldowny - 1]
                self.board_mask[0] = 0
                self.board[0] = 0
                numremove += 1
            else:
                y -= 1
        return numremove               
        
    # 获取可用步骤, 保留一个旋转始终有用
    def set_availables(self):
        if not self.validposition_mask(self.fallpiece, ax = -1):
            self.availables[KEY_LEFT]=0
        else:
            self.availables[KEY_LEFT]=1
        if not self.validposition_mask(self.fallpiece, ax = 1):
            self.availables[KEY_RIGHT]=0
        else:
            self.availables[KEY_RIGHT]=1

        if self.fallpiece['shape']=="o":
            self.availables[KEY_ROTATION]=0
        else:
            r = self.fallpiece['rotation']
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.validposition_mask(self.fallpiece):
                self.availables[KEY_ROTATION]=0
            else:
                self.availables[KEY_ROTATION]=1
            self.fallpiece['rotation'] = r

    # 打印
        
    def step(self, action):                
        # 状态 0 下落过程中 1 更换方块 2 结束一局    
        self.steps += 1
        self.piecesteps += 1

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
                    if not self.validposition_mask(self.fallpiece, ay=1): break

        isFalling=True
        if self.validposition_mask(self.fallpiece, ay=1):
            self.fallpiece['y'] += 1
            if not self.validposition_mask(self.fallpiece, ay=1):
                isFalling = False
        else:
            isFalling = False

        reward = 0
        removedlines = 0
        if not isFalling:
            self.add_piece(self.fallpiece)
            removedlines = self.remove_mask_line()

            self.removedlines += removedlines
            reward = removedlines

            if removedlines>0:
                reward += self.downcount*0.01

                if self.piececount-self.last_reward==1: reward += 1

                self.downcount = 0
                self.last_reward = self.piececount

            self.piecesteps = 0
            self.piececount += 1

            self.fallpiece = self.nextpiece
            self.nextpiece = self.getnewpiece()

            if not self.validposition_mask(self.fallpiece, ay=0):
                self.terminal = True
                removedlines = -1

        # 以下顺序不能变
        self.set_state()
        self.set_availables()

        return removedlines

    def set_state(self):
        self.state[0] = self.board_mask.copy()
        self.state[1] = self.get_fallpiece_board()

    # 打印（字符区分方块，颜色辅助）
    _CH = {0:' ', 1:'S', 2:'Z', 3:'I', 4:'O', 5:'L', 6:'J', 7:'T'}

    def print(self):
        print("")
        board = np.copy(self.board)
        fall_value = PIECE_IDS[self.fallpiece['shape']]
        shapedraw = pieces[self.fallpiece['shape']][self.fallpiece['rotation']]
        for x in range(templatenum):
            for y in range(templatenum):
                if shapedraw[y][x] != blank:
                    w, h = x + self.fallpiece['x'], y + self.fallpiece['y']
                    if 0 <= w < boardwidth and 0 <= h < boardheight:
                        board[h][w] = fall_value

        for y in range(boardheight):
            line = "| "
            for x in range(boardwidth):
                v = int(board[y][x])
                ch = self._CH.get(v, ' ')
                line += ch + " "
            print(line)
        print(" " + " -" * boardwidth)
        print("lines:", self.removedlines, "piececount:", self.piececount, "piecehis:", len(self.piecehis), "/", \
            self.next_Pieces_list_len, "steps:", self.steps)
        print("".join(self.piecehis))


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


    # 获得当前的全部特征
    # 返回 [2, height, width]
    def current_state(self):
        return self.state

