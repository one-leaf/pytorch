import pygame
from pygame.locals import *
import time, sys
from game import pieces, templatenum, blank, black, boardheight, boardwidth, boxsize
from game import colors, white, blue

FPS = 25
winx = 640
winy = 480
xmargin = int(winx-boardwidth*boxsize)/2
topmargin = int(winy-boardheight*boxsize-5)
movedownfreq = 0.1
movesidefreq = 0.15

class TetrominoEnv(object):
    def __init__(self, tetromino):
        self.terminal = tetromino

        pygame.init()
        pygame.display.set_caption('tetromino')
        self.fpsclock = pygame.time.Clock()
        self.disp = pygame.display.set_mode((winx, winy))
        self.bigfont = pygame.font.Font('freesansbold.ttf',100)
        self.basicfont = pygame.font.Font('freesansbold.ttf',20)

    def rungame(self):

        board = self.terminal.getblankboard()
        lastmovedowntime = time.time()
        lastmovesidetime = time.time()
        lastfalltime = time.time()
        movedown = False
        moveleft = False
        moveright = False
        score = 0
        level, fallfreq = self.terminal.calculate(score)
    
        fallpiece = self.terminal.getnewpiece()
        nextpiece = self.terminal.getnewpiece()
    
        while True:
            if fallpiece == None:
                fallpiece = nextpiece
                nextpiece = self.terminal.getnewpiece()
                lastfalltime = time.time()
    
                if not self.terminal.validposition(board,fallpiece):
                    return
                
            self.checkforquit()
            for event in pygame.event.get():
                if event.type == KEYUP:
                    if (event.key == K_p):
                        self.disp.fill(black)
                        # pygame.mixer.music.stop()
                        self.showtextscreen('Paused')
                        # pygame.mixer.music.play(-1,0.0)
                        lastfalltime = time.time()
                        lastmovedowntime = time.time()
                        lastmovesidetime = time.time()
                    elif (event.key == K_LEFT or event.key == K_a):
                        moveleft = False
                    elif (event.key == K_RIGHT or event.key == K_d):
                        moveright = False
                    elif (event.key == K_DOWN or event.key == K_s):
                        movedown = False
                        
                elif event.type == KEYDOWN:
                    if (event.key == K_LEFT or event.key == K_a) and self.terminal.validposition(board,fallpiece,ax = -1):
                        fallpiece['x']-=1
                        moveleft = True
                        moveright = False
                        lastmovesidetime = time.time()
                    elif (event.key == K_RIGHT or event.key == K_d) and self.terminal.validposition(board,fallpiece,ax = 1):
                        fallpiece['x']+=1
                        moveright = True
                        moveleft = False
                        lastmovesidetime = time.time()
    
                    elif (event.key == K_UP or event.key ==K_w):
                        fallpiece['rotation'] =  (fallpiece['rotation'] + 1) % len(pieces[fallpiece['shape']])
                        if not self.terminal.validposition(board,fallpiece):
                            fallpiece['rotation'] = (fallpiece['rotation'] - 1) % len(pieces[fallpiece['shape']])
                    elif (event.key == K_DOWN or event.key ==K_s):
                        movedown = True
                        if self.terminal.validposition(board,fallpiece, ay = 1):
                            fallpiece['y']+=1
                        lastmovedowntime = time.time()
    
                    if event.key == K_SPACE:
                        movedown = False
                        moveleft = False
                        moveright = False
                        for i in range(1,boardheight):
                            if not self.terminal.validposition(board,fallpiece,ay = i):
                                break
                        fallpiece['y'] += i-1
            
            if (moveleft or moveright) and time.time()-lastmovesidetime > movesidefreq:
                if moveleft and self.terminal.validposition(board,fallpiece,ax = -1):
                    fallpiece['x']-=1
                if moveright and self.terminal.validposition(board,fallpiece,ax = 1):
                    fallpiece['x']+=1
                lastmovesidetime = time.time()
    
            if movedown and time.time()-lastmovedowntime>movedownfreq and self.terminal.validposition(board,fallpiece,ay=1):
                fallpiece['y']+=1
                lastmovedowntime = time.time()
            if time.time()-lastfalltime>fallfreq:
                if not self.terminal.validposition(board,fallpiece,ay = 1):
                    self.terminal.addtoboard(board,fallpiece)
                    score +=self.terminal.removecompleteline(board)
                    level,fallfreq = self.terminal.calculate(score)
                    fallpiece = None
                else:
                    fallpiece['y'] +=1
                    lastfalltime = time.time()
    
            self.render(board, score,level, fallpiece, nextpiece)
   
    def render(self, board, score,level, fallpiece, nextpiece):
            self.disp.fill(black)
            self.drawboard(board)
            self.drawstatus(score,level)
            self.drawnextpiece(nextpiece)
            if fallpiece !=None:
                self.drawpiece(fallpiece)
    
            pygame.display.update()
            self.fpsclock.tick(FPS)    

    def terminal(self):
        pygame.quit()
        sys.exit()

    def convertsize(self,boxx,boxy):
        return (boxx*boxsize+xmargin,boxy*boxsize+topmargin)

    def checkforquit(self):
        for event in pygame.event.get(QUIT):
            self.terminal()
        for event in pygame.event.get(KEYUP):
            if event.key == K_ESCAPE:
                self.terminal()
            pygame.event.post(event)
            
    def checkforpress(self):
        self.checkforquit()
        for event in pygame.event.get([KEYDOWN,KEYUP]):
            if event.type == KEYDOWN:
                continue
            return event.key
        return None
    
    def maketext(self,text,font,color):
        surf = font.render(text,1,color)
        return surf,surf.get_rect()
        
    def showtextscreen(self,text):
        tilesurf,tilerect = self.maketext(text,self.bigfont,white)
        tilerect.center = (int(winx/2),int(winy/2))
        self.disp.blit(tilesurf,tilerect)
    
        presssurf,pressrect = self.maketext('press a key to play',self.basicfont,white)
        pressrect.center = (int(winx/2),int(winy/2)+100)
        self.disp.blit(presssurf,pressrect)
    
        while self.checkforpress() == None:
            pygame.display.update()
            self.fpsclock.tick()

    def drawbox(self,boxx,boxy,color,pixelx = None,pixely= None):
        if color == blank:
            return
        if pixelx == None and pixely == None:
            pixelx,pixely = self.convertsize(boxx,boxy)
        pygame.draw.rect(self.disp,colors[color],(pixelx+1 , pixely+1, boxsize-1, boxsize-1))
        
    def drawboard(self,board):
        pygame.draw.rect(self.disp,blue,(xmargin-3,topmargin-7,boardwidth*boxsize+8,boardheight*boxsize+8),5)
        for x in range(boardwidth):
            for y in range(boardheight):
                self.drawbox(x,y,board[x][y])
    
    def drawstatus(self,score,level):
        scoresurf = self.basicfont.render('Score: %s'%score,True,white)
        scorerect = scoresurf.get_rect()
        scorerect.topleft = (winx-150,20)
        self.disp.blit(scoresurf,scorerect)
    
        levelsurf = self.basicfont.render('level: %s'%level,True, white)
        levelrect = levelsurf.get_rect()
        levelrect.topleft = (winx-150,50)
        self.disp.blit(levelsurf,levelrect)
    
    def drawpiece(self,piece,pixelx = None,pixely = None):
        shapedraw = pieces[piece['shape']][piece['rotation']]
        if pixelx == None and pixely == None:
            pixelx,pixely = self.convertsize(piece['x'],piece['y'])
        for x in range(templatenum):
            for y in range(templatenum):
                if shapedraw[y][x]!=blank:
                    self.drawbox(None,None,piece['color'],pixelx+(x*boxsize),pixely + y*boxsize)
    
    def drawnextpiece(self,piece):
        nextsurf = self.basicfont.render('Next:',True,white)
        nextrect =nextsurf.get_rect()
        nextrect.topleft = (winx-120,80)
        self.disp.blit(nextsurf,nextrect)
    
        self.drawpiece(piece,pixelx = winx-120,pixely = 100)

    def main(self):        
        self.showtextscreen('Tetromino')       
        while True:
            # if random.randint(0,1) == 0:
            #     pygame.mixer.music.load('tetrisb.mid')
            # else:
            #     pygame.mixer.music.load('tetrisc.mid')
            # pygame.mixer.music.play(-1,0.0)
            self.rungame()
            # pygame.mixer.music.stop()
            self.showtextscreen('Game Over')