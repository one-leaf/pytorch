import logging
import math
import copy
import random
import numpy as np
import sys
sys.setrecursionlimit(10000)

EPS = 1e-8

log = logging.getLogger(__name__)

# 第二种MCTS实现 s 当前游戏状态， a 当前游戏动作
# 这一种比之前的Tree会省资源，因为 a: [(0,0),(0,1)] 和 a: [(0,1),(0,0)] 是同一种状态，不需要单独进行遍历
class MCTS():
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._policy = policy_value_fn      # 概率估算函数
        self._c_puct = c_puct               # 参数
        self._n_playout = n_playout         # 做几次探索
        self._max_var = 100                # 达到最大方差后停止探索
        self.lable = ""
        self._first_act = set()          # 优先考虑的走法,由于引入了防守奖励，所以不需要优先步骤
        self._limit_max_var = False       # 是否限制最大方差

        self.Qsa = {}  # 保存 Q 值, key: s,a
        self.Nsa = {}  # 保存 遍历次数 key: s,a
        self.Ns = {}  # 保存 遍历次数 key: s
        self.Ps = {}  # 保存 动作概率 key: s, a
        self.Es = {}  # 保存游戏最终得分 key: s
        print("create mcts, c_puct: {}, n_playout: {}".format(c_puct, n_playout))

    def reset(self):
        self.Qsa = {}  # 保存 Q 值, key: s,a
        self.Nsa = {}  # 保存 遍历次数 key: s,a
        self.Ns = {}  # 保存 遍历次数 key: s
        self.Ps = {}  # 保存 动作概率 key: s, a
        self.Es = {}  # 保存游戏最终得分 key: s

    def get_action_probs(self, state, temp=1):
        """
        获得mcts模拟后的最终概率， 输入游戏的当前状态 s
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = state.get_key()

        self.max_depth = 0
        available_acts = state.actions_to_positions(state.availables)
        for n in range(self._n_playout):
            self.depth = 0
            state_copy = copy.deepcopy(state)
            self.search(state_copy)
            if self.depth>self.max_depth: self.max_depth = self.depth

        act_visits = [(a, self.Nsa[(s, a)]) if (s, a) in self.Nsa else (a, 0) for a in available_acts]
        act_Qs = [(a, self.Qsa[(s, a)]) if (s, a) in self.Qsa else (a, 0) for a in available_acts]
        acts = [av[0] for av in act_visits]
        visits = [av[1] for av in act_visits]
        qs = [round(av[1],2) for av in act_Qs]

        if state.state_player == 0:
            info=[]
            for idx in sorted(range(len(visits)), key=visits.__getitem__)[::-1]:
                if len(info)>2: break
                act, visit = act_visits[idx]
                action = state.position_to_action(act)
                q, p= 0,0
                if (s, act) in self.Qsa: q = self.Qsa[(s, act)]
                if s in self.Ps: p = self.Ps[s][act]
                info.append([action, visit, round(q,2), round(p,2)]) 
            print(state.steps, state.piececount, state.piecesteps, self.lable, s ,"n_playout:", n, "depth:" ,self.max_depth, info)
                #, \   "first:", state.positions_to_actions(list(self._first_act)[-3:]))

        if temp == 0:
            bestAs = np.array(np.argwhere(visits == np.max(visits))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(visits)
            probs[bestA] = 1
            return acts, np.array(probs), qs

        m = np.power(np.array(visits), 1./temp)
        m_sum = np.sum(m)
        if m_sum<=0:
            v_len = len(acts)
            probs = np.ones(v_len)/v_len
        else:
            probs = m/m_sum
        return acts, probs, qs

    def search(self, state):
        """
        蒙特卡洛树搜索        
        NOTE: 返回当前局面的状态 [-1,1] 如果是当前玩家是 v ，如果是对手是 -v.
        返回:
            v: 当前局面的状态
        """
        s = state.get_key(True)
        self.depth = self.depth +1

        # 将所有状态的得分都 cache 起来
        if s not in self.Es:
            end, _ = state.game_end()  
            v = 0
            # 这里是对上一步的评价，如果游戏结束对我而言都是不利的，v为-1
            # 这里增加了最终的奖励，提升对步骤的优化
            # if (end and winner != -1): 
                # if state.current_player==winner:
                #     v = 1 * 2
                # else:
                # v = -1 * 1
            # 这个有问题，计算量有点大，放弃
            # elif state.check_will_win():
            #     v = -2

            # 游戏结束双方都输掉了
            if end:
                ph=state.pieces_height
                ph_max=max(ph)

                if ph[-1]>=ph_max:
                    v = 1
                else:
                    v = -1
                if state.state_player == 0:
                    v = -1 * v
                # print("end",state.state_player,ph_avg,ph)
                # if state.state_player == 0 :
                #     if ph[-1]==max(ph):
                #         v = -1 
                #     else:
                #         v = 1
                # else:
                #     if ph[-1]>ph_avg:
                #         v = -1
                #     else:
                #         v = 1 
                # if state.state_player==0:
                #     v = max_height/10 - 1
                # else:
                #     v = 1 - max_height/10

            # 谁先落下来谁赢，这样由于下落的可能高，就倒逼正常走子远离下落
            # if state.state==1:
            #     max_height = state.fallpiece_height
            #     k=0.01
            #     if state.state_player==0:
            #         v =  k * (1 - (max_height/20))  
            #     else:
            #         v =  k * (1 - (max_height/20)) * -1 #-0.01 / max_height             
            #     # if state.reward==0:
            #     #     v =  0.1/max_height
            #     # else:
            #     #     v = -0.1/max_height

            

            self.Es[s] = v

        # 如果得分不等于0，标志这局游戏结束
        if self.Es[s] != 0 or state.terminal:
            return -self.Es[s]

        # 如果当前状态没有子节点，增加子节点
        # 增加 Ps[s] Vs[s] Ns[s]
        if s not in self.Ps:               
            # 获得当前局面的概率 和 局面的打分, 这个已经过滤掉了不可用走子
            act_probs, v = self._policy(state)
            probs = np.zeros(state.actions_num)
            for act, prob in act_probs:
                probs[act] = prob
            self.Ps[s] = probs 

            self.Ns[s] = 0

            return -v

        # 当前最佳概率和最佳动作
        cur_best = -float('inf')
        best_act = -1

        # 如果动作 a 在优先探索步骤内，且没有从来没有被探索过，则优先探索
        # 训练后期不需要了
        # for a in state.actions_to_positions(state.availables):
        #     if a in self._first_act and not (s, a) in self.Qsa: 
        #         best_act = a
        #         break

        if best_act == -1:
            # 选择具有最高置信上限的动作
            for a in state.actions_to_positions(state.availables):
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self._c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self._c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # 加一个EPS小量防止 Q = 0 

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        act = state.position_to_action(a)
        state.step(act)

        # 计算下一步的 v 这个v 为正数，但下一个v为负数
        if state.reward >0:
            v = -1.
            print("!",end="")   
            if state.state_player == 1:
                v = -1 * v        
        elif state.state==1: 
            ph=state.pieces_height
            # avg_ph = sum(ph)/len(ph)
            # v = (-1./avg_ph)
            # v = avg_ph/20 不能用这个，这个会导致粘连
            ph_max=max(ph)
            if ph[-1]>=ph_max:
                v = 0.1
            else:
                v = -0.1
            if state.state_player == 1:
                v = -1 * v    
            v = v + self.search(state)
        else:
            v = self.search(state)

        # 更新 Q 值 和 访问次数
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    # 返回某个局面的action信息
    def getInfo(self, state, act):
        s = state
        n = q = p = 0
        if (s, act) in self.Nsa: n = self.Nsa[(s, act)]
        if (s, act) in self.Qsa: q = self.Qsa[(s, act)]
        if s in self.Ps: p = self.Ps[s][act]
        return n, q, p

class MCTSPurePlayer(object):
    """基于纯MCTS的player"""

    @staticmethod
    def policy_value_fn(state):
        """给棋盘所有可落子位置分配默认平均概率 [(0, 0.015625), (action, probability), ...], 0"""
        availables = state.actions_to_positions(state.availables)
        action_probs = np.ones(len(availables)) / len(availables)

        # 返回的 v 采用快速走子的办法
        limit = 1000 # 最大探测次数
        winner = -1
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action = random.choice(state.availables)
            state.step(action)
        v = 0
        if winner != -1:  # 如果不是平局
            v = 1 if winner != state.current_player else -1

        return  [(availables[i], action_probs[i]) for i in range(len(availables))], v

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(MCTSPurePlayer.policy_value_fn, c_puct, n_playout)
        self.mcts._limit_max_var = False
        # self.mcts._max_var = 300

    def set_player_ind(self, p):
        """指定MCTS的playerid"""
        self.player = p
        self.mcts.lable = "MCTS(%s)"%p

    def reset_player(self):
        self.mcts.reset()

    def get_action(self, state, return_prob=0):
        """计算下一步走子action"""
        if len(state.availables) > 0:  # 盘面可落子位置>0
            # 构建纯MCTS初始树(节点分布充分)，并返回child中访问量最大的action
            acts, act_probs, act_qs = self.mcts.get_action_probs(state, temp=0)

            move_probs = np.zeros(state.actions_num)
            move_probs[acts] = act_probs

            idx = np.argmax(act_probs) 
            act = acts[idx]

            # 第一步棋为一手交换，随便下
            if state.step_count==0: 
                action = random.choice(state.first_availables)
                act =  state.action_to_position(action)
            else:
                action = state.position_to_action(act)

            if act!=acts[idx]:
                print("    random:", state.position_to_action(acts[idx]), act_probs[idx], act_qs[idx], \
                     "==>", action, act_probs[acts.index(act)], act_qs[acts.index(act)])

            if return_prob:
                return action, move_probs
            else:
                return action
        else:
            print("WARNING: the state is full")

    def __str__(self):
        return "MCTS {}".format(self.player)


class MCTSPlayer(object):
    """基于模型指导概率的MCTS + AI player"""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        """初始化参数"""
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        """指定MCTS的playerid"""
        self.player = p
        self.mcts.lable = "AI(%s)"%p

    def reset_player(self):
        self.mcts.reset()

    def get_action(self, state, temp=0, return_prob=0, return_value=0):        
        """计算下一步走子action"""
        move_probs = np.zeros(state.actions_num)
        value = 0
        if not state.terminal:  # 如果游戏没有结束
            # 训练的时候 temp = 1
            acts, act_probs, act_qs = self.mcts.get_action_probs(state, temp)
            move_probs[acts] = act_probs
            idx = np.argmax(act_probs)    

            if self._is_selfplay:  # 自我对抗
                act = acts[idx]
                # action = state.position_to_action(act)
                value = act_qs[idx]

                # 如果标准差低于0.02，可以随机
                # if act_qs[idx]>0 and state.step_count<state.n_in_row:
                # print(np.std(act_probs))              
                # if np.std(act_probs)<0.02 : 
                # 早期多随机
                if state.piecesteps<20:
                    p = 0.5                 
                    dirichlet = np.random.dirichlet(0.03 * np.ones(len(act_probs)))
                    act = np.random.choice(acts, p=p * act_probs + (1.0-p) * dirichlet)
                action = state.position_to_action(act)

                if act!=acts[idx]:
                    print("    random:", state.position_to_action(acts[idx]), act_probs[idx], act_qs[idx], \
                        "==>", action, act_probs[acts.index(act)], act_qs[acts.index(act)])  
                                                                  
            else:  # 和人类对战
                act = acts[idx]
                action = state.position_to_action(act)
                value = act_qs[idx]

                # 如果盘面看好，可以随机
                # if act_qs[idx]>0 and state.step_count<state.n_in_row: 
                #     act = np.random.choice(acts, p=act_probs)
                #     action = state.position_to_action(act)

                if act!=acts[idx]:
                    print("    random:", state.position_to_action(acts[idx]), act_probs[idx], act_qs[idx], \
                        "==>", action, act_probs[acts.index(act)], act_qs[acts.index(act)])  
            if return_prob:
                return action, move_probs
            elif return_value:
                return action, value        
            else:
                return action
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "AI {}".format(self.player)