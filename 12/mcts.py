import logging
import math
import copy
import random
import numpy as np

EPS = 1e-8

class MCTS():
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
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
        print("create mcts, c_puct: {}, n_playout: {}".format(c_puct, n_playout))

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

        if state.state != 0:
            info=[]
            for idx in sorted(range(len(visits)), key=visits.__getitem__)[::-1]:
                act, visit = act_visits[idx]
                action = state.position_to_action_name(act)
                q, p = 0,0
                if (s, act) in self.Qsa: q = self.Qsa[(s, act)]
                if s in self.Ps: p = self.Ps[s][act]
                info.append([action, visit, round(q,2), round(p,2)])        
            v = 0
            if s in self.Vs: v = self.Vs[s]
            print(state.steps, state.piecesteps, state.piececount, "n:", n, "depth:" ,self.max_depth, "value:", round(v,2), info)

        if temp == 0:
            bestAs = np.array(np.argwhere(visits == np.max(visits))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(visits)
            probs[bestA] = 1
            return acts, np.array(probs), qs

        # print(visits, temp)
        # m = np.power(np.array(visits), 1./temp)
        m = np.array(visits)
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
        s = state.get_key()
        self.depth = self.depth +1

        # 将所有状态的得分都 cache 起来
        # 由于不确定最后结果，所以按预测的为准
        if s not in self.Es:
            v = 0
            if state.terminal:
               _, v = self._policy(state)
            self.Es[s] = v

        # 如果得分不等于0，标志这局游戏结束
        if self.Es[s] != 0 or state.terminal:
            return self.Es[s]

        # 如果当前状态没有子节点，增加子节点
        # 增加 Ps[s] Vs[s] Ns[s]
        if s not in self.Ps:                          
            # 获得当前局面的概率 和 局面的打分, 这个已经过滤掉了不可用走法
            act_probs, v = self._policy(state)
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
        state.step(act)
        
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

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000):
        """初始化参数"""
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)

    def set_player_ind(self, p):
        """指定MCTS的playerid"""
        self.player = p
        self.mcts.lable = "AI(%s)"%p

    def reset_player(self):
        self.mcts.reset()

    def get_action(self, state, temp=0, return_prob=0, return_value=0, need_random=True):        
        """计算下一步走子action"""
        move_probs = np.zeros(state.actions_num)
        value = 0
        if not state.terminal:  # 如果游戏没有结束
            # 训练的时候 temp = 1
            acts, act_probs, act_qs = self.mcts.get_action_probs(state, temp)
            move_probs[acts] = act_probs
            idx = np.argmax(act_probs)    

            if need_random:  # 自我对抗
                act = acts[idx]
                value = act_qs[idx]

                # 早期多随机
                # if act in [0,4] and random.random()>0.5:
                # if act_probs[idx]<0.99:
                # if abs(value)>0.5 or random.random()>0.95:
                # if state.piececount < 50 and (state.piecesteps<3 or value<-0.9):
                p = 0.75                 
                dirichlet = np.random.dirichlet(0.03 * np.ones(len(act_probs)))
                act = np.random.choice(acts, p=p * act_probs + (1.0-p) * dirichlet)
                action = state.position_to_action(act)

                if act!=acts[idx]:
                    print("    random:", state.position_to_action_name(acts[idx]), "==>", state.position_to_action_name(act))  
                                                                  
            else:  # 和人类对战
                act = acts[idx]
                action = state.position_to_action(act)
                value = act_qs[idx]

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
            print("WARNING: game is terminal")

    def __str__(self):
        return "AI {}".format(self.player)