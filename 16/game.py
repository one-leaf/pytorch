import random
 
boxsize = 20
boardwidth = 10
boardheight = 20

templatenum = 5
 
white = (255,255,255)
black = (0,0,0)
blue = (0,0,255)
yellow = (255,255,0)
green = (0,255,0)
purple = (255,0,255)
red = (255,0,0)
cyan = (0,255,255)
blank = '.'
colors = (yellow,green,purple,red,cyan,(128,128,0),(0,128,0),(128,0,128),(128,0,0),(0,128,128))
 
stemplate = [['.....',
              '..00.',
              '.00..',
              '.....',
              '.....'],
             ['.....',
              '..o..',
              '..00.',
              '...0.',
              '.....']]
 
ztemplate = [['.....',
              '.00..',
              '..00.',
              '.....',
              '.....'],
             ['.....',
              '...0.',
              '..00.',
              '..0..',
              '.....']]
 
itemplate = [['..0..',
              '..0..',
              '..0..',
              '..0..',
              '.....'],
             ['.....',
              '.0000',
              '.....',
              '.....',
              '.....']]
 
otemplate = [['.....',
              '..00.',
              '..00.',
              '.....',
              '.....']]
 
ltemplate = [['.....',
              '..0..',
              '..0..',
              '..00.',
              '.....'],
             ['.....',
              '...0.',
              '.000.',
              '.....',
              '.....'],
             ['.....',
              '..00.',
              '...0.',
              '...0.',
              '.....'],
             ['.....',
              '.000.',
              '.0...',
              '.....',
              '.....']]
 
jtemplate = [['.....',
              '..0..',
              '..0..',
              '.00..',
              '.....'],
             ['.....',
              '.000.',
              '...0.',
              '.....',
              '.....'],
             ['.....',
              '..00.',
              '..0..',
              '..0..',
              '.....'],
             ['.....',
              '.0...',
              '.000.',
              '.....',
              '.....']]
 
ttemplate = [['.....',
              '..0..',
              '.000.',
              '.....',
              '.....'],
             ['..0..',
              '.00..',
              '..0..',
              '.....',
              '.....'],
             ['.....',
              '.000.',
              '..0..',
              '.....',
              '.....'],
             ['..0..',
              '..00.',
              '..0..',
              '.....',
              '.....']]
pieces = {'s':stemplate,
          'z':ztemplate,
          'i':itemplate,
          'o':otemplate,
          'l':ltemplate,
          'j':jtemplate,
          't':ttemplate}


# 本次下落的方块中点地板的距离
def landingHeight(piece):
    shape=pieces[piece['shape']][piece['rotation']]
    for y in range(templatenum):
        for x in range(templatenum):
            if shape[x][y] != blank:
                return boardheight - (piece['y'] + y)

# 本次下落后此方块贡献（参与完整行组成的个数）*完整行的行数
def rowsEliminated(board, piece):
    eliminatedNum = 0
    eliminatedGridNum = 0
    shape=pieces[piece['shape']][piece['rotation']]
    for y in range(boardheight):
        flag = True
        for x in range(boardwidth):
            if board[x][y] == blank:
                flag = False
                break
        if flag:
            eliminatedNum += 1
            if (y>piece['y']) and (y <piece['y']+templatenum):
                for s in range(templatenum):
                    if shape[y-piece['y']][s] != blank:
                            eliminatedGridNum += 1
    return eliminatedNum * eliminatedGridNum

# 在同一行，方块 从无到有 或 从有到无 算一次（边界算有方块）
def rowTransitions(board):
    totalTransNum = 0
    for y in range(boardheight):
        nowTransNum = 0
        currisBlank = False
        for x in range(boardwidth):
            isBlank = board[x][y] == blank
            if currisBlank != isBlank:
                nowTransNum += 1
                currisBlank = isBlank
        if currisBlank:   
            nowTransNum += 1
        totalTransNum += nowTransNum
    return totalTransNum  

# 在同一列，方块 从无到有 或 从有到无 算一次（边界算有方块）
def colTransitions(board):
    totalTransNum = 0
    for x in range(boardwidth):
        nowTransNum = 0
        currisBlank = False
        for y in range(boardheight):
            isBlank = board[x][y] == blank
            if currisBlank != isBlank:
                nowTransNum += 1
                currisBlank = isBlank
        if  currisBlank:   
            nowTransNum += 1
        totalTransNum += nowTransNum
    return totalTransNum   

# 空洞的数量。空洞无论有多大，只算一个。一个图中可能有多个空洞
def emptyHoles(board):
    totalEmptyHoles = 0
    for x in range(boardwidth):
        y = 0
        emptyHoles = 0
        while y < boardheight:
            if board[x][y]!=blank:
                y += 1
                break
            y += 1 
        while y < boardheight:
            if board[x][y]==blank:
                emptyHoles += 1
            y += 1
        totalEmptyHoles += emptyHoles
    return totalEmptyHoles

# 井就是两边都有方块的空列。（空洞也可以是井，一列中可能有多个井）。此值为所有的井以1为公差首项为1的等差数列的总和
def wellNums(board):
    totalWellDepth  = 0
    wellDepth = 0
    tDepth = 0
    # 获取左边的井数
    for y in range(boardheight):            
        if board[0][y] == blank and board[1][y] != blank:
            tDepth += 1
        else:
            wellDepth += tDepth * (tDepth+1) / 2    
            tDepth = 0
    wellDepth += tDepth * (tDepth+1) / 2  
    totalWellDepth += wellDepth
    # 获取中间的井数
    wellDepth = 0.
    for x in range(1,boardwidth-1):
        tDepth = 0.
        for y in range(boardheight):
            if board[x][y]==blank and board[x-1][y]!=blank and board[x+1][y]!=blank:
                tDepth += 1
            else:
                wellDepth += tDepth * (tDepth+1) / 2
                tDepth = 0
        wellDepth += tDepth * (tDepth+1) / 2
    totalWellDepth += wellDepth
    # 获取最右边的井数
    wellDepth = 0
    tDepth = 0
    for y in range(boardheight):
        if board[boardwidth-1][y] == blank and board[boardwidth-2][y] != blank:
            tDepth += 1
        else:
            wellDepth += tDepth * (tDepth +1 )/2
            tDepth = 0
    wellDepth += tDepth * (tDepth +1 )/2
    totalWellDepth += wellDepth
    return totalWellDepth        

# 修改了价值评估 下落高度 消行个数 行变化次数 列变化次数 空洞个数 井的个数
def calcReward(board, piece):
    _landingHeight = landingHeight(piece)
    _rowsEliminated = rowsEliminated(board, piece)
    _rowTransitions = rowTransitions(board)
    _colTransitions = colTransitions(board)
    _emptyHoles = emptyHoles(board)
    _wellNums = wellNums(board)
    return -4.500158825082766 * _landingHeight \
                + 3.4181268101392694 * _rowsEliminated \
                + -3.2178882868487753 * _rowTransitions \
                + -9.348695305445199 * _colTransitions \
                + -7.899265427351652 * _emptyHoles \
                + -3.3855972247263626 * _wellNums; 

class Tetromino(object):  
    def __init__(self, isRandomNextPiece=True):
        self.nextpiece=[]
        self.isRandomNextPiece=isRandomNextPiece
        self.pieceCount = 0

    def getrandompiece(self):
        shape = random.choice(list(pieces.keys()))
        newpiece = {'shape':shape,
                    'rotation': 0,
                    'x': int(boardwidth)//2-int(templatenum//2),
                    'y': -1,
                    'color': 0}
        return newpiece

    def calculate(self,score):
        level = int(score/10)+1
        fallfreq = 0.27-(level*0.02)
        return level,fallfreq
        
    def getnewpiece(self):
        if not self.isRandomNextPiece:
            if len(self.nextpiece)<100:
                for i in range(99):
                    self.nextpiece.insert(0, self.getrandompiece())                    
            nextpiece = self.nextpiece.pop()
        else:
            nextpiece = self.getrandompiece()  
        nextpiece["color"] = self.pieceCount % len(colors)  
        self.pieceCount += 1
        return nextpiece
    
    def getblankboard(self):
        board = []
        for x in range(boardwidth):
            board.append([blank]*boardheight)
        return board
    
    def addtoboard(self,board,piece):
        for x in range(templatenum):
            for y in range(templatenum):
                w = x + piece['x']
                h = y + piece['y']
                if pieces[piece['shape']][piece['rotation']][y][x]!=blank:
                    if w>=0 and w<boardwidth and h>=0 and h<boardheight:
                        board[w][h] = piece['color']
                
    def onboard(self,x,y):
        return x >=0 and x<boardwidth and y<boardheight
        
    def validposition(self,board,piece,ax = 0,ay = 0):
        for x in range(templatenum):
            for y in range(templatenum):
                aboveboard = y +piece['y'] +ay < 0
                if aboveboard or pieces[piece['shape']][piece['rotation']][y][x]== blank:
                    continue
                if not self.onboard(x + piece['x']+ax,y+piece['y']+ay):
                    return False
                # print(piece['x'],piece['y'])
                if board[x+piece['x']+ax][y+piece['y']+ay]!=blank:
                    return False
        return True
    
    
    def completeline(self,board,y):
        for x in range(boardwidth):
            if board[x][y]==blank:
                return False
        return True
    
    def removecompleteline(self,board):
        numremove = 0
        y = boardheight-1
        while y >=0:
            if self.completeline(board,y):
                for pulldowny in range(y,0,-1):
                    for x in range (boardwidth):
                        board[x][pulldowny] = board[x][pulldowny-1]
                for x in range(boardwidth):
                    board[x][0] = blank
                numremove+=1
            else:
                y-=1
        return numremove
    

if __name__ == '__main__':
    tetromino = Tetromino()
    from gameenv import TetrominoEnv
    env = TetrominoEnv(tetromino)
    env.main()