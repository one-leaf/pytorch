import gym
import random
from gym.envs.classic_control import rendering
import time

class FiveChess(object):
    def __init__(self, size=15, n_in_row=5):
        # 棋盘大小
        self.size = size
        # 最少多少子连线才算赢
        self.n_in_row = n_in_row
        # 初始棋盘是0    -1表示黑棋子   1表示白棋子
        self.chessboard = [ [  0 for v in range(self.size)  ] for v in range(self.size) ]
        self.step_count = 0
        self.players = [0, 1]
        self.colors = [1, -1]
        # 可用步骤
        self.reset()

    def reset(self):
        self.chessboard = [ [  0 for v in range(self.size)  ] for v in range(self.size) ]
        self.step_count = 0
        self.last_action = None
        self.current_player = self.players[0]
        self.availables = [(x,y) for x in range(self.size) for y in range(self.size)]
        return self.chessboard
 
    # 检查当前action是否有效
    def is_valid_set_coord(self, action):
        return action in self.availables
 
    # 返回所有有效的下棋位置
    def get_available_locations(self):
        results = []
        for x in range(self.size):
            for y in range(self.size):
                if self.chessboard[x][y]==0:
                    results.append([x,y])
        return results
 
    # 检查是否游戏结束,返回赢的用户0 或 1，如果平局返回-1
    def check_terminal(self):
        # 如果都没有下子的位置了，则返回平局
        if len(self.get_available_locations())==0:
            return True, -1

        # 遍历落子位置，检查是否出现横/竖/斜线上n子相连的情况
        n = self.n_in_row
        for x in range(self.size):
            for y in range(self.size):
                color = self.chessboard[x][y]
                if color == 0: continue
                if self.size-x>=n and abs(sum([self.chessboard[x+i][y] for i in range(n)]))==n:  
                    return True, self.colors.index(color)

                if self.size-y>=n and abs(sum([self.chessboard[x][y+i] for i in range(n)]))==n:  
                    return True, self.colors.index(color)

                if self.size-x>=n and self.size-y>=n and abs(sum([self.chessboard[x+i][y+i] for i in range(n)]))==n:  
                    return True, self.colors.index(color)

                if self.size-x>=n and y>=n and abs(sum([self.chessboard[x+i][y-i] for i in range(n)]))==n:  
                    return True, self.colors.index(color)
        return False, -1        

    #action 包括坐标和  例如：[1,3] 表示： 坐标（1,3）
    #输出 下一个状态，动作价值，是否结束，赢的用户
    def step(self, action):
        if action not in self.availables:
            print(action)
            raise "action error"  
        self.last_action = action     
        self.availables.remove(action)
        self.step_count +=1
        self.current_player = self.step_count % 2

        #胜负判定
        color = self.colors[self.current_player]
        #棋子
        self.chessboard[action[0]][action[1]] = color
        terminal, user = self.check_terminal()
        reward = 0 if user==-1 else 1 
        return self.chessboard, reward, terminal, {"user":user}

class FiveChessEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, fiveChess):
        self.fiveChess = fiveChess
        self.viewer = None

    def reset(self):
        self.fiveChess.reset()

    def step(self, action):
        return self.fiveChess.step(action)

    def render(self, mode = 'human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
 
        screen_width = 800
        screen_height = 800
        space = 50
        width = (screen_width - space*2)/(self.fiveChess.size-1)
 
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            bg = rendering.FilledPolygon([(0,0),(screen_width,0),(screen_width,screen_height),(0,screen_height),(0,0)])
            bg.set_color(0.2,0.2,0.2)
            self.viewer.add_geom(bg)
            #棋盘网格
            for i in range(self.fiveChess.size):
                line = rendering.Line((space,space+i*width),(screen_width-space,space+i*width))
                line.set_color(1, 1, 1)
                self.viewer.add_geom(line)
            for i in range(self.fiveChess.size):
                line = rendering.Line((space+i*width,space),(space+i*width,screen_height - space))
                line.set_color(1, 1, 1)
                self.viewer.add_geom(line)
                
            #棋子
            self.chess = []
            for x in range(self.fiveChess.size):
                self.chess.append([])
                for y in range(self.fiveChess.size):
                    c = rendering.make_circle(width/3)
                    ct = rendering.Transform(translation=(0,0))
                    c.add_attr(ct)
                    c.set_color(0,0,0)
                    self.chess[x].append([c,ct])
                    self.viewer.add_geom(c)

        for x in range(self.fiveChess.size):
            for y in range(self.fiveChess.size):	
                if self.fiveChess.chessboard[x][y]!=0:
                    self.chess[x][y][1].set_translation(space+x*width,space+y*width)
                    if self.fiveChess.chessboard[x][y]==1:
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
        w = 700./(self.fiveChess.size-1)
        for i in range(0, self.fiveChess.size):
            if x > (i-0.5) * w  and x < (i+0.5)*w:
                ax = i 
                break
        for i in range(0, self.fiveChess.size):
            if y > (i-0.5) * w and y < (i+0.5)*w:
                ay = i 
                break
        return (ax, ay)

if __name__ == "__main__":
    user_point = None
    def on_mouse_press(x, y, button, modifiers):
        global user_point
        user_point = (x, y)
    fiveChess = FiveChess(size=15, n_in_row=5)
    env = FiveChessEnv(fiveChess)

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
                    if env.fiveChess.is_valid_set_coord(action):
                        user_point = None
                        break
                env.render()
                time.sleep(0.1)
            _, reward, done, info = env.step(action)
            env.render(mode="human",close=False)
        else:
            available_locations = env.fiveChess.availables
            action = random.choice(available_locations)
            _, reward, done, info = env.step(action)
            env.render(mode="human",close=False)
        
        is_human = not is_human
        
        if done:
            curr_user = env.current_player
            step_count = env.step_count
            print(f'win_user: {info["user"]}, curr_user: {curr_user} reward: {reward} step_count: {step_count}')

    env.close()            
