import gym
import random
from gym.envs.classic_control import rendering

import time
 
class FiveChess(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
 
    def __init__(self, size=15, n_in_row=5):
        # 棋盘大小
        self.SIZE = size
        # 最少多少子连线才算赢
        self.n_in_row = n_in_row
        # 初始棋盘是0    -1表示黑棋子   1表示白棋子
        self.chessboard = [ [  0 for v in range(self.SIZE)  ] for v in range(self.SIZE) ]
        self.viewer = None
        self.step_count = 0
        self.players = [1, -1]
        self.reset()
        
    def is_valid_coord(self,x,y):
        return x>=0 and x<self.SIZE and y>=0 and y<self.SIZE
 
    def is_valid_set_coord(self,x,y):
        return self.is_valid_coord(x,y) and self.chessboard[x][y]==0
 
    #返回有效的下棋位置
    def get_available_locations(self):
        results = []
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if self.chessboard[x][y]==0:
                    results.append([x,y])
        return results
 
    # 获取当前用户
    def get_current_player(self):
        return self.step_count%2

    # 检查是否游戏结束,返回赢的用户0 或 1，如果平局返回-1
    def check_terminal(self):
        # 如果都没有下子的位置了，则返回平局
        if len(self.get_available_locations())==0:
            return True, -1

        # 遍历落子位置，检查是否出现横/竖/斜线上n子相连的情况
        n = self.n_in_row
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                color = self.chessboard[x][y]
                if color == 0: continue
                if self.SIZE-x>=n and abs(sum([self.chessboard[x+i][y] for i in range(n)]))==n:  
                    return True, self.players.index(color)

                if self.SIZE-y>=n and abs(sum([self.chessboard[x][y+i] for i in range(n)]))==n:  
                    return True, self.players.index(color)

                if self.SIZE-x>=n and self.SIZE-y>=n and abs(sum([self.chessboard[x+i][y+i] for i in range(n)]))==n:  
                    return True, self.players.index(color)

                if self.SIZE-x>=n and y>=n and abs(sum([self.chessboard[x+i][y-i] for i in range(n)]))==n:  
                    return True, self.players.index(color)
        return False, -1        

    #action 包括坐标和  例如：[1,3] 表示： 坐标（1,3）
    #输出 下一个状态，动作价值，是否结束，赢的用户
    def step(self, action):
        self.step_count +=1
        #胜负判定
        color = self.players[self.get_current_player()]
        #棋子
        self.chessboard[action[0]][action[1]] = color
        terminal, user = self.check_terminal()
        reward = 0 if user==-1 else 1 
        return self.chessboard,reward,terminal,{"user":user}

    def reset(self):
        self.chessboard = [ [  0 for v in range(self.SIZE)  ] for v in range(self.SIZE) ]
        self.step_count = 0
        self.current_player = self.players[0]
        return self.chessboard
 
    def render(self, mode = 'human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
 
        screen_width = 800
        screen_height = 800
        space = 50
        width = (screen_width - space*2)/(self.SIZE-1)
 
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            bg = rendering.FilledPolygon([(0,0),(screen_width,0),(screen_width,screen_height),(0,screen_height),(0,0)])
            bg.set_color(0.2,0.2,0.2)
            self.viewer.add_geom(bg)
            
            #棋盘网格
            for i in range(self.SIZE):
                line = rendering.Line((space,space+i*width),(screen_width-space,space+i*width))
                line.set_color(1, 1, 1)
                self.viewer.add_geom(line)
            for i in range(self.SIZE):
                line = rendering.Line((space+i*width,space),(space+i*width,screen_height - space))
                line.set_color(1, 1, 1)
                self.viewer.add_geom(line)
                
            #棋子
            self.chess = []
            for x in range(self.SIZE):
                self.chess.append([])
                for y in range(self.SIZE):
                    c = rendering.make_circle(width/3)
                    ct = rendering.Transform(translation=(0,0))
                    c.add_attr(ct)
                    c.set_color(0,0,0)
                    self.chess[x].append([c,ct])
                    self.viewer.add_geom(c)

        for x in range(self.SIZE):
            for y in range(self.SIZE):	
                if self.chessboard[x][y]!=0:
                    self.chess[x][y][1].set_translation(space+x*width,space+y*width)
                    if self.chessboard[x][y]==1:
                        self.chess[x][y][0].set_color(255,255,255)
                    else:
                        self.chess[x][y][0].set_color(0,0,0)
                else:
                    self.chess[x][y][1].set_translation(-10,-10)
 
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

if __name__ == "__main__":
    env = FiveChess(size=10,n_in_row=3)
    while True:
        env.reset()
        done = False
        while not done:
            available_locations = env.get_available_locations()
            
            action = random.choice(available_locations)

            _, reward, done, info = env.step(action)
            env.render(True)

            if done:
                curr_user = env.get_current_player()
                step_count = env.step_count
                print(f'win_user: {info["user"]}, curr_user: {curr_user} reward: {reward} step_count: {step_count}')
                time.sleep(5)
                break

