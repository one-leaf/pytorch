import gym
import random
from gym.envs.classic_control import rendering
import time
import numpy as np
from numpy.lib.stride_tricks import broadcast_arrays

class FiveChessEnv(gym.Env):
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
        # 可用步骤
        self.reset()

    def reset(self):
        self.chessboard = [ [  0 for v in range(self.SIZE)  ] for v in range(self.SIZE) ]
        self.step_count = 0
        self.last_action = None
        self.current_player = self.players[0]
        self.availables = [(x,y) for x in range(self.SIZE) for y in range(self.SIZE)]
        return self.chessboard
 
    # 检查当前action是否有效
    def is_valid_set_coord(self, action):
        return action in self.availables
 
    # 返回所有有效的下棋位置
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
        if action not in self.availables: raise "action error"  
        self.last_action = action     
        self.availables.remove(action)
        self.step_count +=1
        #胜负判定
        color = self.players[self.get_current_player()]
        #棋子
        self.chessboard[action[0]][action[1]] = color
        terminal, user = self.check_terminal()
        reward = 0 if user==-1 else 1 
        return self.chessboard, reward, terminal, {"user":user}
 
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

    # 鼠标点击坐标转为动作坐标
    def point_to_action(self, point):
        x, y = point
        x = x - 50
        y = y - 50
        ax = ay = 0
        w = 700./(self.SIZE-1)
        for i in range(0, self.SIZE):
            if x > (i-0.5) * w  and x < (i+0.5)*w:
                ax = i 
                break
        for i in range(0, self.SIZE):
            if y > (i-0.5) * w and y < (i+0.5)*w:
                ay = i 
                break
        return (ax, ay)

class Agent(object):
    def __init__(self, fivechess):
        self.env = fivechess
        self.env.reset()

    def do_move(self, action):
        self.env.step(action)

    def game_end(self):
        return self.env.check_terminal()

    def get_availables(self):
        return self.env.availables

    # 返回 [1, 4, size, size]
    def current_state(self):
        square_state = np.zeros((4, self.size, self.size))
        board = self.env.chessboard
        curr_player_id = self.env.get_current_player()
        # 前面2层是自己和对手的棋
        for x in range(self.size):
            for y in range(self.size):
                if board[x][y]!=0:
                    idx = 0 if board[x][y]==self.env.players[curr_player_id] else 1
                    square_state[idx][x][y] = 1.0
        # 第三层为最后一步
        x,y = self.env.last_action
        square_state[2][x][y] = 1.0
        # 第四层为如果当前用户是先手则为1
        if curr_player_id == 0:
            square_state[3][:,:] = 1.0
        return square_state

    def start_self_play(self):
        pass

if __name__ == "__main__":
    user_point = None
    def on_mouse_press(x, y, button, modifiers):
        global user_point
        user_point = (x, y)

    env = FiveChessEnv(size=15, n_in_row=5)
    while True:
        env.reset()
        env.render()
        env.viewer.window.on_mouse_press = on_mouse_press
        done = False

        is_human = True        
        while not done:
            if is_human:
                while True:
                    if user_point!=None:
                        action = env.point_to_action(user_point)
                        if env.is_valid_set_coord(action):
                            user_point = None
                            break
                    env.render()
                    time.sleep(0.1)
                _, reward, done, info = env.step(action)
                env.render(mode="human",close=False)
            else:
                available_locations = env.availables
                action = random.choice(available_locations)
                _, reward, done, info = env.step(action)
                env.render(mode="human",close=False)
            
            is_human = not is_human
            
            if done:
                curr_user = env.get_current_player()
                step_count = env.step_count
                print(f'win_user: {info["user"]}, curr_user: {curr_user} reward: {reward} step_count: {step_count}')
                for i in range(50):
                    time.sleep(0.1)
                break

