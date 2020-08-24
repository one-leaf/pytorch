from game import Tetromino, pieces, templatenum, boardwidth, boardheight, blank, black
import copy
import pygame
from pygame.locals import *
import sys

KEY_ROTATION  = [0,1,0]
KEY_LEFT      = [1,0,0]
KEY_RIGHT     = [0,0,1]

# 本次下落的方块中点地板的距离
def landingHeight(board, piece):
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
def calcReward(tetromino, board, piece):
    _landingHeight = landingHeight(board, piece)
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

# 在游戏开始就计算多有的可能分值
def calcAllRewards(tetromino, board, piece):
    rewards=[]
    rotationCount = len(pieces[piece['shape']]) 
    maxReward=-10000
    x_reward=0
    r_reward=0
    for r in range(rotationCount):
        m_piece = copy.deepcopy(piece)  
        m_piece['rotation']=r
        for x in range(boardwidth+10):
            m_board =  copy.deepcopy(board)
            m_piece['x']=x-5                
            for y in range(boardheight+10):
                m_piece['y']=y-1  
                if not tetromino.validposition(m_board, m_piece):
                    continue

                if not tetromino.validposition(m_board, m_piece, ay = 1):
                    tetromino.addtoboard(m_board,m_piece)
                    reward = calcReward(tetromino, m_board, m_piece)
                    if reward > maxReward :
                        maxReward = reward
                        x_reward=m_piece['x']
                        r_reward=r
                    rewards.append(reward)
                    break
    # print(rewards,r_reward,x_reward )                        
    return rewards,r_reward,x_reward

class Agent(object):
    def __init__(self, tetromino):
        self.tetromino = tetromino
        self.reset()

    def reset(self):
        self.fallpiece = self.tetromino.getnewpiece()
        self.nextpiece = self.tetromino.getnewpiece()
        self.score = 0
        self.level = 0
        self.board = self.tetromino.getblankboard()
        self.calc_reward = 0.0 
        self.rewards, self.reward_r, self.reward_x = calcAllRewards(self.tetromino, self.board, self.fallpiece)      
    
    def step(self, action):
        reward  = 0
        moveleft = False
        moveright = False
        is_terminal = False
        shape  = self.fallpiece['shape']
        self.level = self.tetromino.calculate(self.score)

        if action == KEY_LEFT and self.tetromino.validposition(self.board,self.fallpiece,ax = -1):
            self.fallpiece['x']-=1
            moveleft = True

        if action == KEY_RIGHT and self.tetromino.validposition(self.board,self.fallpiece,ax = 1):
            self.fallpiece['x']+=1  
            moveright = True 

        if action == KEY_ROTATION:
            self.fallpiece['rotation'] =  (self.fallpiece['rotation'] + 1) % len(pieces[self.fallpiece['shape']])
            if not self.tetromino.validposition(self.board,self.fallpiece):
                self.fallpiece['rotation'] = (self.fallpiece['rotation'] - 1) % len(pieces[self.fallpiece['shape']])

        if not self.tetromino.validposition(self.board,self.fallpiece,ay = 1):
            self.tetromino.addtoboard(self.board,self.fallpiece)
            reward = calcReward(self.tetromino, self.board, self.fallpiece)
            self.score += self.tetromino.removecompleteline(self.board)            
            self.level = self.tetromino.calculate(self.score)   

            self.fallpiece = None
        else:
            self.fallpiece['y'] +=1

        self.tetromino.disp.fill(black)
        self.tetromino.drawboard(self.board)
        self.tetromino.drawstatus(self.score, str(self.level))
        self.tetromino.drawnextpiece(self.nextpiece)
        if self.fallpiece !=None:
            self.tetromino.drawpiece(self.fallpiece)

        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()

        if self.fallpiece == None:
            self.fallpiece = self.nextpiece
            self.nextpiece = self.tetromino.getnewpiece()
            if not self.tetromino.validposition(self.board,self.fallpiece):   
                is_terminal = True       
                self.reset()     
                return reward, screen_image, is_terminal, shape, self.rewards  # 虽然游戏结束了，但还是正常返回分值，而不是返回 -1
            self.rewards, self.reward_r, self.reward_x  = calcAllRewards(self.tetromino, self.board, self.fallpiece) # 计算下一步最佳分值
        return reward, screen_image, is_terminal, shape, self.rewards

    def autoStep(self):
        if self.fallpiece['rotation']!=self.reward_r:
            self.step(KEY_ROTATION)
            return
        if self.fallpiece['x']>self.reward_x:
            self.step(KEY_LEFT)
            return
        if self.fallpiece['x']<self.reward_x:
            self.step(KEY_RIGHT)
            return
        self.step(None)

def main(agent):
    while True:
        # time.sleep(1)
        for event in pygame.event.get():  # 需要事件循环，否则白屏
            if event.type == QUIT:
                pygame.quit()
                sys.exit()    

        agent.autoStep()
        

if __name__ == "__main__":
    tetromino = Tetromino()
    agent = Agent(tetromino)
    main(agent)