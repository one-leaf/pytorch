from typing import Tuple
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

    def reset(self, start_player=0, need_shuffle_availables=True):
        self.chessboard = [ [  0 for v in range(self.size)  ] for v in range(self.size) ]
        self.step_count = 0
        self.current_player = self.players[start_player]
        availables = [(x,y) for x in range(self.size) for y in range(self.size)]
        # 按照先中间后两边的排序
        availables = sorted(availables, key=lambda x : (x[0]-self.size//2)**2+(x[1]-self.size//2)**2)
        # 一手交换的第一步棋
        self.first_availables = availables[self.size*self.size//3:self.size*self.size*2//3]
        # 随机打乱位置
        if need_shuffle_availables:
            random.shuffle(availables)
        self.availables=availables

        self.terminal = False
        self.win_user = -1
        self.actions=[]
        return self.chessboard

    # 返回棋局的唯一key
    def get_key(self):
        key = [0 for v in range(self.size*self.size)]
        for x in range(self.size):
            for y in range(self.size):
                key[x+y*self.size]=str(self.chessboard[x][y]+1)
        key3 = int("".join(key),3)
        return hash(key3)

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
        if len(self.availables)==0:
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

    # 检查是否是防守成功
    def is_defend(self, action=None, debug=False):
        if self.step_count<self.n_in_row*2:
            return False
        n = self.n_in_row
        _chessboard = copy.deepcopy(self.chessboard)
        if action == None:
            last_x, last_y = self.actions[-1]
        else:
            last_x, last_y = action
            color = self.colors[self.current_player]
            _chessboard[last_x][last_y] = color

        c = _chessboard[last_x][last_y]
        _chessboard[last_x][last_y]= -1 if c==1 else 1
        isend,_ = self.check_terminal(_chessboard, (last_x, last_y))
        if isend: 
            if debug:
                print("win", (last_x, last_y))
                for y in range(self.size-1, -1, -1):
                    line="%s "%(y%10)
                    for x in range(self.size):
                        char = " "
                        if _chessboard[x][y]==1:
                            char = "X"
                        if _chessboard[x][y]==-1:
                            char = "O"
                        line += char+" "
                    print(line)
                print("  "+str.join(" ",[str(i%10) for i in range(self.size)]))
            return True
        will_win = self.check_will_win(_chessboard, (last_x, last_y))
        if will_win:
            if debug:
                print("will win", (last_x, last_y))
                for y in range(self.size-1, -1, -1):
                    line="%s "%(y%10)
                    for x in range(self.size):
                        char = " "
                        if _chessboard[x][y]==1:
                            char = "X"
                        if _chessboard[x][y]==-1:
                            char = "O"
                        line += char+" "
                    print(line)
                print("  "+str.join(" ",[str(i%10) for i in range(self.size)]))    
            return True
        return False            
        


    # 检查是否是四个子且两端都是空白位置，并且剩余的都没有一步可以赢的
    def check_will_win(self, chessboard=None, action=None):
        if self.step_count<self.n_in_row*2:
            return False

        if chessboard==None:
            _chessboard=self.chessboard
        else:
            _chessboard=chessboard
        if action==None:
            last_x, last_y = self.actions[-1]
        else:
            last_x, last_y = action

        n = self.n_in_row
        c = _chessboard[last_x][last_y]

        hassame=1
        curr_search_pass=True
        for l in range(1, n):
            curr_x = last_x+l
            if curr_x==self.size or (_chessboard[curr_x][last_y]!=c and _chessboard[curr_x][last_y]!=0):
                curr_search_pass=False
                break 
            if _chessboard[curr_x][last_y]==0: break 
            hassame += 1
        if curr_search_pass:
            for l in range(1, n):
                curr_x = last_x-l
                if curr_x<0 or (_chessboard[curr_x][last_y]!=c and _chessboard[curr_x][last_y]!=0):
                    curr_search_pass=False
                    break
                if _chessboard[curr_x][last_y]==0: break
                hassame += 1
        if curr_search_pass and hassame>=n-1: return True
        
        hassame=1
        curr_search_pass=True
        for l in range(1, n):
            curr_y = last_y+l
            if curr_y==self.size or (_chessboard[last_x][curr_y]!=c and _chessboard[last_x][curr_y]!=0):
                curr_search_pass=False
                break 
            if _chessboard[last_x][curr_y]==0: break 
            hassame += 1
        if curr_search_pass:
            for l in range(1, n):
                curr_y = last_y-l
                if curr_y<0 or (_chessboard[last_x][curr_y]!=c and _chessboard[last_x][curr_y]!=0):
                    curr_search_pass=False
                    break
                if _chessboard[last_x][curr_y]==0: break
                hassame += 1
        if curr_search_pass and hassame>=n-1: return True

        hassame=1
        curr_search_pass=True
        for l in range(1, n):
            curr_x = last_x+l
            curr_y = last_y+l
            if curr_x == self.size or curr_y==self.size or (_chessboard[curr_x][curr_y]!=c and _chessboard[curr_x][curr_y]!=0):
                curr_search_pass=False
                break 
            if _chessboard[curr_x][curr_y]==0: break 
            hassame += 1
        if curr_search_pass:
            for l in range(1, n):
                curr_x = last_x-l
                curr_y = last_y-l
                if curr_x<0 or curr_y<0 or (_chessboard[curr_x][curr_y]!=c and _chessboard[curr_x][curr_y]!=0):
                    curr_search_pass=False
                    break
                if _chessboard[curr_x][curr_y]==0: break
                hassame += 1
        if curr_search_pass and hassame>=n-1: return True

        hassame=1
        curr_search_pass=True
        for l in range(1, n):
            curr_x = last_x-l
            curr_y = last_y+l
            if curr_x <0 or curr_y==self.size or (_chessboard[curr_x][curr_y]!=c and _chessboard[curr_x][curr_y]!=0):
                curr_search_pass=False
                break 
            if _chessboard[curr_x][curr_y]==0: break 
            hassame += 1
        if curr_search_pass:
            for l in range(1, n):
                curr_x = last_x+l
                curr_y = last_y-l
                if curr_x==self.size or curr_y<0 or (_chessboard[curr_x][curr_y]!=c and _chessboard[curr_x][curr_y]!=0):
                    curr_search_pass=False
                    break
                if _chessboard[curr_x][curr_y]==0: break
                hassame += 1
        if curr_search_pass and hassame>=n-1: return True
        return False

    # 检查是否游戏结束,返回赢的用户0 或 1，如果平局返回-1
    def check_terminal(self, chessboard=None, action=None):
        # 如果都没有足够的棋
        if self.step_count<self.n_in_row*2-1:
            return False, -1
        # 如果都没有下子的位置了，或，则返回平局
        if len(self.availables)==0:
            return True, -1

        if chessboard==None:
            _chessboard=self.chessboard
        else:
            _chessboard=chessboard

        # 找到最后一个子
        if action ==None:
            last_x, last_y = self.actions[-1]
        else:
            last_x, last_y = action
            
        n = self.n_in_row
        c = _chessboard[last_x][last_y]
        lastplayer = self.players[0] if self.current_player == self.players[1] else self.players[1]

        hassame=1
        for l in range(1, n):
            if last_x+l==self.size or _chessboard[last_x+l][last_y]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        for l in range(1, n):
            if last_x-l<0 or _chessboard[last_x-l][last_y]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        
        hassame=1
        for l in range(1, n):
            if last_y+l==self.size or _chessboard[last_x][last_y+l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        for l in range(1, n):
            if last_y-l<0 or _chessboard[last_x][last_y-l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer

        hassame=1
        for l in range(1, n):
            if last_x+l==self.size or last_y+l==self.size or _chessboard[last_x+l][last_y+l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        for l in range(1, n):
            if last_x-l<0 or last_y-l<0 or _chessboard[last_x-l][last_y-l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer

        hassame=1
        for l in range(1, n):
            if last_x-l<0 or last_y+l==self.size or _chessboard[last_x-l][last_y+l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer
        for l in range(1, n):
            if last_x+l==self.size or last_y-l<0 or _chessboard[last_x+l][last_y-l]!=c: break
            hassame += 1
        if hassame>=n: return True, lastplayer

        return False, -1   

    #action 包括坐标和  例如：[1,3] 表示： 坐标（1,3）
    #输出 下一个状态，动作价值，是否结束，赢的用户
    def step(self, action):
        if action not in self.availables:
            print(action)
            print(self.availables)
            raise "action error, action not in availables"  
        self.availables.remove(action)
        self.actions.append(action)
        self.step_count +=1

        #棋子
        color = self.colors[self.current_player]
        self.chessboard[action[0]][action[1]] = color

        #这一步完成
        self.current_player = self.players[0] if self.current_player==self.players[1] else self.players[1]

        #胜负判定
        self.terminal, self.win_user = self.check_terminal()
        reward = 0 if self.win_user==-1 else 1 

        return self.chessboard, reward, self.terminal, self.win_user

    # 概率的索引位置转action
    def position_to_action(self, position):
        return (position%self.size, self.size-(position//self.size)-1)

    def positions_to_actions(self, positions):
        return [self.position_to_action(i) for i in positions]

    # action转概率的索引位置
    def action_to_position(self, action):
        x,y = action
        return x+(self.size-y-1)*self.size

    def actions_to_positions(self, actions):
        return [self.action_to_position(act) for act in actions]

    # 返回 [1, 11, size, size]
    # alphago zero使用了17层即 [1,17,19,19] 的网络，前8层为当前玩家的最后八步，后8层为对手的最后八步，最后一层是当前玩家是否为先手，如果先手则全部为1
    # 因此这里我们采用前5层自己的最后5步棋，后5层为对手的最后5步棋，最后一层自己是否是先手
    def current_state(self):
        square_state = np.zeros((11, self.size, self.size))
        # 前面8层是自己和对手的棋包括最后三步的棋
        # 由于最后一步始终是对手的棋，所以倒序后，始终为 
        # step: 9,7,5,3,1,8,6,4,2,0
        #  idx: 0,1,2,3,4,5,6,7,8,9
        for i, act in enumerate(self.actions[::-1]):
            if i%2==0: 
                idx=[5,6,7,8,9]
            else:
                idx=[0,1,2,3,4]
            if i == 0: idx=[9]
            if i == 1: idx=[4]
            if i == 2: idx=[8,9]
            if i == 3: idx=[3,4]
            if i == 4: idx=[7,8,9]
            if i == 5: idx=[2,3,4]
            if i == 6: idx=[6,7,8,9]
            if i == 7: idx=[1,2,3,4]
            x,y = act
            for j in idx:
                square_state[j,self.size-y-1,x] = 1.0

        # 第11层为如果当前用户是先手则为1
        if self.step_count % 2 == 1:
            square_state[-1][:,:] = 1.0

        return square_state

    # 打印状态
    def print(self):
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
        print("win:",self.win_user, "curr:", self.current_player, "is_first:", self.step_count % 2 == 1)
        print("actions:", self.actions)

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
    fiveChess = FiveChess(size=15, n_in_row=5)
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
