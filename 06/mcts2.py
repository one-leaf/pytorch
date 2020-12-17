import logging
import math
import copy

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)

# 第二种MCTS实现 s 当前游戏状态， a 当前游戏动作
# 这一种比之前的Tree会省资源，因为 a: [(0,0),(0,1)] 和 a: [(0,1),(0,0)] 是同一种状态，不需要单独进行遍历
class MCTS():
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._policy = policy_value_fn      # 概率估算函数
        self._c_puct = c_puct               # 参数
        self._n_playout = n_playout         # 做几次探索

        self.Qsa = {}  # 保存 Q 值, key: s,a
        self.Nsa = {}  # 保存 遍历次数 key: s,a
        self.Ns = {}  # 保存 遍历次数 key: s
        self.Ps = {}  # 保存 动作概率 key: s, a

        self.Es = {}  # 保存游戏最终得分 key: s
        self.Vs = {}  # 保存游戏可用步骤 key: s

    def get_action_probs(self, state, temp=1):
        """
        获得mcts模拟后的最终概率， 输入游戏的当前状态 s
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = state.get_key()

        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self.search(state_copy)
            
            if n >= len(state.availables):

                # 取出当前所有的 visits
                visits = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in state.availables]

                if len(visits)==1: break
                var = np.var(visits)
                if var>self._max_var: break

        act_visits = [(a, self.Nsa[(s, a)]) if (s, a) in self.Nsa else 0 for a in state.availables]
        acts = [av[0] for av in act_visits]
        visits = [av[1] for av in act_visits]

        if temp == 0:
            bestAs = np.array(np.argwhere(visits == np.max(visits))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(visits)
            probs[bestA] = 1
            return acts, probs

        visits = [x ** (1. / temp) for x in visits]
        counts_sum = float(sum(visits))
        probs = [x / counts_sum for x in visits]
        return acts, probs

    def search(self, state):
        """
        蒙特卡洛树搜索        
        NOTE: 返回当前局面的状态 [-1,1] 如果是当前玩家是 v ，如果是对手是 -v.
        返回:
            v: 当前局面的状态
        """
        s = state.get_key()

        # 将所有状态的得分都 cache 起来
        if s not in self.Es:
            end, winner = state.game_end()  
            v = 0
            if end:
                if state.current_player==winner:
                    v = 1
                else:
                    v = -1                         
            self.Es[s] = v

        # 如果得分不等于0，标志这局游戏结束
        if self.Es[s] != 0:
            return -self.Es[s]

        # 如果当前状态没有子节点，增加子节点
        # 增加 Ps[s] Vs[s] Ns[s]
        if s not in self.Ps:
            # 获得当前局面的概率 和 局面的打分, 这个是全量的
            act_probs, v = self._policy(state)

            # 得到当前所有可用动作对应的向量位置
            positions = state.actions_to_positions(state.availables)
            valids = np.zeros(state.size * state.size)
            for p in positions:
                valids[p]=1.
            
            self.Ps[s] = act_probs * valids

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # 归一化
            else:                
                log.error("当前所有可用走子概率的和为0")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]

        # 当前最佳概率和最佳动作
        cur_best = -float('inf')
        best_act = -1

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
        act = state.positions_to_actions([a])[0]
        state.step(act)

        # 计算下一步的 v 这个v 为正数，但下一个v为负数
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