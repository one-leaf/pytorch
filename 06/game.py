import gym
import random
from gym.envs.classic_control import rendering
import time
import numpy as np
import copy

class FiveChess(object):
    def __init__(self, size=15, n_in_row=5):
        # 棋盘大小
        self.size = size
        # 最少多少子连线才算赢
        self.n_in_row = n_in_row
        self.chessboard = [ [  0 for v in range(self.size)  ] for v in range(self.size) ]
        self.step_count = 0
        self.players = [0, 1]
        # 初始棋盘是0    -1表示黑棋子   1表示白棋子
        self.colors = [-1, 1]
        self.reset()

    def reset(self, start_player=0):
        self.chessboard = [ [  0 for v in range(self.size)  ] for v in range(self.size) ]
        self.step_count = 0
        self.current_player = self.players[start_player]
        availables = [(x,y) for x in range(self.size) for y in range(self.size)]
        # 按照先中间后两边的排序
        self.availables = sorted(availables, key=lambda x : (x[0]-self.size//2)**2+(x[1]-self.size//2)**2)
        self.terminal = False
        self.win_user = -1
        self.players_actions=[[],[]]
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
    def check_terminal2(self):
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

    # 检查是否游戏结束,返回赢的用户0 或 1，如果平局返回-1
    def check_terminal(self):
        # 如果都没有足够的棋
        if self.step_count<self.n_in_row*2-1:
            return False, -1
        # 如果都没有下子的位置了，或，则返回平局
        if len(self.get_available_locations())==0:
            return True, -1

        # 找到最后一个子
        lastplayer = self.players[0] if self.current_player==self.players[1] else self.players[1]
        last_x, last_y = self.players_actions[lastplayer][-1]
        n = self.n_in_row
        c = self.chessboard[last_x][last_y]

        hassame=1
        for l in range(1, n):
            if last_x+l==self.size or self.chessboard[last_x+l][last_y]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        for l in range(1, n):
            if last_x-l<0 or self.chessboard[last_x-l][last_y]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        
        hassame=1
        for l in range(1, n):
            if last_y+l==self.size or self.chessboard[last_x][last_y+l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        for l in range(1, n):
            if last_y-l<0 or self.chessboard[last_x][last_y-l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer

        hassame=1
        for l in range(1, n):
            if last_x+l==self.size or last_y+l==self.size or self.chessboard[last_x+l][last_y+l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        for l in range(1, n):
            if last_x-l<0 or last_y-l<0 or self.chessboard[last_x-l][last_y-l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer

        hassame=1
        for l in range(1, n):
            if last_x-l<0 or last_y+l==self.size or self.chessboard[last_x-l][last_y+l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        for l in range(1, n):
            if last_x+l==self.size or last_y-l<0 or self.chessboard[last_x+l][last_y-l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer

        return False, -1   

    #action 包括坐标和  例如：[1,3] 表示： 坐标（1,3）
    #输出 下一个状态，动作价值，是否结束，赢的用户
    def step(self, action):
        if action not in self.availables:
            print(action)
            raise "action error"  
        self.availables.remove(action)

        #棋子
        color = self.colors[self.current_player]
        self.chessboard[action[0]][action[1]] = color
        self.players_actions[self.current_player].append(action)

        #这一步完成
        self.step_count +=1
        self.current_player = self.players[0] if self.current_player==self.players[1] else self.players[1]

        #胜负判定
        self.terminal, self.win_user = self.check_terminal()
        reward = 0 if self.win_user==-1 else 1 

        return self.chessboard, reward, self.terminal, self.win_user

    # 概率的索引位置转action
    def positions_to_actions(self, positions):
        return [(i%self.size, self.size-(i//self.size)-1) for i in positions]

    # action转概率的索引位置
    def actions_to_positions(self, actions):
        return [x+(self.size-y-1)*self.size for x,y in actions]

    # 返回 [1, 7, size, size]
    def current_state(self):
        square_state = np.zeros((7, self.size, self.size))
        # 前面6层是自己和对手的棋包括最后三步的棋
        for x in range(self.size):
            for y in range(self.size):
                if self.chessboard[x][y]!=0:
                    # 检测这个棋是否在最后三步
                    player = self.players[self.colors.index(self.chessboard[x][y])]
                    action = (x,y)
                    last_actions = self.players_actions[player][-3:]
                    if player == self.current_player:
                        idx = 0 
                    else:
                        idx = 3    
                    if action in last_actions:
                        idx +=  last_actions.index(action)
                    square_state[idx,self.size-y-1,x] = 1.0     

        # 第四层为如果当前用户是先手则为1
        if self.step_count % 2 == 0:
            square_state[6][:,:] = 1.0

        # 归一化数据
        square_state = (square_state - 0.5) / 0.5
        return square_state

    # 获得当前棋和下一次的尝试的模拟走法截图
    # 一次最多取16笔进行返回
    def current_and_next_state(self):
        availables = []
        max_len = len(self.availables)
        if max_len > 15:
            max_len = 15
        square_state = np.zeros((max_len+1, 7, self.size, self.size))
        square_state[0] = self.current_state()
        availables.append(self.availables)
        for i, ac in enumerate(self.availables[:max_len]):
            game = copy.deepcopy(self)
            game.step(ac)
            square_state[i+1] = game.current_state()
            availables.append(game.availables)
        return square_state, availables

    # 打印状态
    def print(self, state=None):
        if not state is None:
            state = state*0.5+0.5
            state1 = state[0:3].sum(0)
            state2 = state[3:6].sum(0)
            for y in range(self.size-1, -1, -1):
                line="%s "%(y%10)
                for x in range(self.size):
                    char = " "
                    if state1[x][y]==1:
                        char = "X"
                    if state2[x][y]==1:
                        char = "O"
                    line += char+" "
                print(line)
            print("  "+str.join(" ",[str(i%10) for i in range(self.size)]))
        else:
            for y in range(self.size-1, -1, -1):
                line="%s "%(y%10)
                for x in range(self.size):
                    char = " "
                    if self.chessboard[x][y]==1:
                        char = "X"
                    if self.chessboard[x][y]==-1:
                        char = "O"
                    line += char+" "
                print(line)
            print("  "+str.join(" ",[str(i%10) for i in range(self.size)]))
        print("currr_player:", self.current_player, "is_first:", self.step_count % 2 == 0, "last_action:",self.players_actions[0][-1],self.players_actions[1][-1])

    def game_end(self):
        return self.terminal, self.win_user

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
                    c.set_color(0, 0, 0)
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
                    self.chess[x][y][1].set_translation(-1*width,-1*width)

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
    fiveChess = FiveChess(size=8, n_in_row=5)
    env = FiveChessEnv(fiveChess)

    env.reset()
    # fiveChess.step((2,3))
    # print(fiveChess.chessboard)
    # print(fiveChess.current_state())
    # # env.render()
    # # time.sleep(5)
    # raise "x"

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
            _, reward, done, win_user = env.step(action)
            env.render(mode="human",close=False)
        else:
            available_locations = env.fiveChess.availables
            action = random.choice(available_locations)
            _, reward, done, win_user = env.step(action)
            env.render(mode="human",close=False)
        
        # print(fiveChess.current_state())
        # print("-----------------")

        is_human = not is_human
        
        if done:
            curr_user = fiveChess.current_player
            step_count = fiveChess.step_count
            print(f'win_user: {win_user}, curr_user: {curr_user} reward: {reward} step_count: {step_count}')

        if env.fiveChess.step_count>2:
            env.fiveChess.print()
    env.close()            
