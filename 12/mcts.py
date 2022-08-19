from itertools import count
import logging
import math
import copy
import random
import numpy as np

EPS = 1e-8

class MCTS():
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, flip_v=False):
        self._policy = policy_value_fn      # 概率估算函数
        self._c_puct = c_puct               # 参数
        self._n_playout = n_playout         # 做几次探索
        self.lable = ""
        self._first_act = set()          # 优先考虑的走法,由于引入了防守奖励，所以不需要优先步骤

        self.Qsa = {}  # 保存 Q 值, key: s,a
        self.Nsa = {}  # 保存 遍历次数 key: s,a
        self.Ns = {}  # 保存 遍历次数 key: s
        self.Ps = {}  # 保存 动作概率 key: s, a
        self.Es = {}  # 保存游戏最终得分 key: s
        self.Vs = {}  # 保存游戏局面打分 key: s # 这个不需要，只是缓存
        print("create mcts, c_puct: {}, n_playout: {}, flip_v: {}".format(c_puct, n_playout, flip_v))

        self.state = None
        self.flip_v = flip_v

    def reset(self):
        self.Qsa = {}  # 保存 Q 值, key: s,a
        self.Nsa = {}  # 保存 遍历次数 key: s,a
        self.Ns = {}  # 保存 遍历次数 key: s
        self.Ps = {}  # 保存 动作概率 key: s, a
        self.Es = {}  # 保存游戏最终得分 key: s
        self.Vs = {}  # 保存游戏局面打分 key: s # 这个不需要，只是缓存

    def get_action_probs(self, state, temp=1):
        """
        获得mcts模拟后的最终概率， 输入游戏的当前状态 s
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = state.get_key()
        self.state = state
        self.max_depth = 0
        available_acts = state.actions_to_positions(state.availables)
        # for n in range(self._n_playout):
        for n in count():
            self.depth = 0
            state_copy = copy.deepcopy(state)
            self.search(state_copy)
            if self.depth>self.max_depth: self.max_depth = self.depth

            # 如果只有一种走法，只探测一次
            if len(available_acts)==1 : break

            # 当前状态
            v = self.Vs[s] if s in self.Vs else 0
            visits_sum = sum([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in available_acts])          
            # 如果当前状态不错，总次数 200
            # if v>0 and visits_sum >= 200 :break

            # 如果当前状态不佳，总次数 2000
            if visits_sum>=2000 or state.terminal: break

            # 保底，防止莫名其妙的问题
            if n >= self._n_playout : break

            # if state.pieceheight<3 and visits_sum >= self._n_playout*20: break
            # if n >= self._n_playout and v > 0 and visits_sum >= self._n_playout*10: break

        act_visits = [(a, self.Nsa[(s, a)]) if (s, a) in self.Nsa else (a, 0) for a in available_acts]
        act_Qs = [(a, self.Qsa[(s, a)]) if (s, a) in self.Qsa else (a, 0) for a in available_acts]
        acts = [av[0] for av in act_visits]
        visits = [av[1] for av in act_visits]
        qs = [round(av[1],2) for av in act_Qs]
        v = 0 if s not in self.Vs else self.Vs[s]

        if state.show_mcts_process or state.pieceheight in [0,9] :
            info=[]
            for idx in sorted(range(len(visits)), key=visits.__getitem__)[::-1]:
                act, visit = act_visits[idx]
                action = state.position_to_action_name(act)
                q, p = 0,0
                if (s, act) in self.Qsa: q = self.Qsa[(s, act)]
                if s in self.Ps: p = self.Ps[s][act]
                info.append([action, visit, round(q,2), round(p,2)])        
            print(state.steps, state.piececount, state.fallpiece["shape"], state.piecesteps, "n:", n, "depth:" ,self.max_depth,"height:", state.pieceheight, "value:", round(v,2), info, "flip_v:", self.flip_v)

        if temp == 0:
            bestAs = np.array(np.argwhere(visits == np.max(visits))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(visits)
            probs[bestA] = 1
            return acts, np.array(probs), qs, v

        m = np.power(np.array(visits), 1./temp)
        m_sum = np.sum(m)
        if m_sum<=0:
            v_len = len(acts)
            probs = np.ones(v_len)/v_len
        else:
            probs = m/m_sum

        return acts, probs, qs, v

    def search(self, state):
        """
        蒙特卡洛树搜索        
        NOTE: 返回当前局面的状态 [-1,1] 如果是当前玩家是 v ，如果是对手是 -v.
        返回:
            v: 当前局面的状态
        """
        s = state.get_key()
        self.depth = self.depth +1

        if self.depth>1000: return 0

        if state.terminal: self.Es[s] = -1

        # 如果得分不等于0，标志探索结束
        if s in self.Es: return self.Es[s]

        # 如果当前状态没有子节点，增加子节点
        # 增加 Ps[s] Vs[s] Ns[s]
        if s not in self.Ps:                          
            # 获得当前局面的概率 和 局面的打分, 这个已经过滤掉了不可用走法
            act_probs, v = self._policy(state)
            if self.flip_v: v = -v
            probs = np.zeros(state.actions_num)
            for act, prob in act_probs:
                probs[act] = prob
            self.Ps[s] = probs 

            self.Ns[s] = 0
            self.Vs[s] = v

            return v

        # 当前最佳概率和最佳动作
        cur_best = -float('inf')
        best_act = -1

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
        prev_pieceheight = state.pieceheight

        state.step(act)
        
        # 局部奖励
        # off = (prev_pieceheight - state.pieceheight) * state.pieceheight/10.
        # v  = off + self.search(state)
        v = self.search(state)

        # 更新 Q 值 和 访问次数
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v

    # 返回某个局面的action信息
    def getInfo(self, state, act):
        s = state
        n = q = p = 0
        if (s, act) in self.Nsa: n = self.Nsa[(s, act)]
        if (s, act) in self.Qsa: q = self.Qsa[(s, act)]
        if s in self.Ps: p = self.Ps[s][act]
        return n, q, p

class MCTSPlayer(object):
    """基于模型指导概率的MCTS + AI player"""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, flip_v=False):
        """初始化参数"""
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, flip_v)

    def set_player_ind(self, p):
        """指定MCTS的playerid"""
        self.player = p
        self.mcts.lable = "AI(%s)"%p

    def reset_player(self):
        self.mcts.reset()

    def get_action(self, state, temp=0):        
        """计算下一步走子action"""
        move_probs = np.zeros(state.actions_num)
        value = 0
        if not state.terminal:  # 如果游戏没有结束
            # 训练的时候 temp = 1
            acts, act_probs, act_qs, state_v = self.mcts.get_action_probs(state, temp)
            move_probs[acts] = act_probs
            max_idx = np.argmax(act_probs)    

            if temp==0:
                idx = max_idx
            else:
                p = 0.9                 
                dirichlet = np.random.dirichlet(0.03 * np.ones(len(act_probs)))
                idx = np.random.choice(range(len(acts)), p=p * act_probs + (1.0-p) * dirichlet)                                                                     

            # if state.pieceheight>=4:
            #     qs_idx = np.argmax(act_qs)
            #     if idx!=qs_idx and act_probs[qs_idx]>0.2:
            #         idx =  qs_idx

            action = acts[idx]
            value = act_qs[idx]

            # if idx!=max_idx:
            #     move_probs[idx], move_probs[max_idx] = move_probs[max_idx], move_probs[idx]

            if state.show_mcts_process:
                if idx!=max_idx:
                    print("    random:", state.position_to_action_name(acts[max_idx]), act_probs[max_idx], "==>", state.position_to_action_name(acts[idx]), act_probs[idx])  

            # print(acts, act_probs, idx, action)

            return action, move_probs, state_v
        else:
            print("WARNING: game is terminal")

    def __str__(self):
        return "AI {}".format(self.player)