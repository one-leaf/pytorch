from itertools import count
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

        self.state = None

    def reset(self):
        self.Qsa = {}  # 保存 Q 值, key: s,a
        self.Nsa = {}  # 保存 遍历次数 key: s,a
        self.Ns = {}  # 保存 遍历次数 key: s
        self.Ps = {}  # 保存 动作概率 key: s, a
        self.Es = {}  # 保存游戏最终得分 key: s
        self.Vs = {}  # 保存游戏局面打分 key: s # 这个不需要，只是缓存

    def get_action_probs(self, game, temp=1):
        """
        获得mcts模拟后的最终概率， 输入游戏的当前状态 s
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        s = game.get_key()
        self.max_depth = 0
        available_acts = game.availables
        game.piececount_mark = game.piececount
        # for n in range(self._n_playout):
        for n in count():
            self.depth = 0
            game_ = copy.deepcopy(game)
            # for i, g in enumerate(games_dict["games"]):
            #     print(i, [a['shape'] for a in g.tetromino.nextpiece[-20:]])
            self.search(game_)
            if self.depth>self.max_depth: self.max_depth = self.depth

            # 如果只有一种走法，只探测一次
            if game.terminal: break
            if len(available_acts)==1 : break
            if self.depth > 800 : break
            # if self.depth < 200 and n < self._n_playout*10: continue 

            # 当前状态
            # v = self.Vs[s] if s in self.Vs else 0
            # visits_sum = sum([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in available_acts])          
            # act_visits = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in available_acts]
            # act_Qs = [self.Qsa[(s, a)] if (s, a) in self.Qsa else 0 for a in available_acts]
            # max_qs = max(act_Qs)

            # 这样网络不稳定
            # if game.piececount==0 and visits_sum>128: break
            # if np.argmax(act_Qs)==np.argmax(act_visits) and visits_sum > 2048: break
            # 如果探索总次数大于2048次就别探索了。
            # if visits_sum>=2048 or game.terminal: break

            # 如果达到最大探索次数，结束探索
            if n >= self._n_playout -1 : break

        act_visits = [(a, self.Nsa[(s, a)]) if (s, a) in self.Nsa else (a, 0) for a in available_acts]
        act_Qs = [(a, self.Qsa[(s, a)]) if (s, a) in self.Qsa else (a, 0) for a in available_acts]
        acts = [av[0] for av in act_visits]
        visits = [av[1] for av in act_visits]
        qs = [av[1] for av in act_Qs]
        ps = [self.Ps[s][a] if s in self.Ps else 0 for a in available_acts]
        v = 0 if s not in self.Vs else self.Vs[s]
        ns = 1 if s not in self.Ns else self.Ns[s]
        if ns>temp and temp>0:
            temp = np.log(ns)/np.log(temp)
        else:
            temp = 1

        if temp == 0:
            bestAs = np.array(np.argwhere(visits == np.max(visits))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(visits)
            probs[bestA] = 1
            probs = np.array(probs)
        else:
            temp = 1/temp
            m = np.power(np.array(visits), temp)
            m_sum = np.sum(m)
            if m_sum<=0:
                v_len = len(acts)
                probs = np.ones(v_len)/v_len
            else:
                probs = m/m_sum

        qval = qs[np.argmax(probs)]

        if game.show_mcts_process or game.state == 1 :
            info=[]
            for idx in sorted(range(len(visits)), key=visits.__getitem__)[::-1]:
                act, visit = act_visits[idx]
                q, p = 0,0
                if (s, act) in self.Qsa: q = self.Qsa[(s, act)]
                if s in self.Ps: p = self.Ps[s][act]
                info.append([game.position_to_action_name(act), visit, round(q,2), round(p,2)])        
            print(game.steps, game.fallpiece["shape"], "temp:", round(temp,2), "ns:", ns, "depth:", self.max_depth, "empty:", game.getEmptyCount(), "height:", round(game.pieceheight,2), "value:", round(v,2), "qval:", round(qval,2), info, "player:", game.id)

        return acts, probs, qs, ps, v, ns

    def search(self, game):
        """
        蒙特卡洛树搜索        
        NOTE: 返回当前局面的状态 [-1,1] 如果是当前玩家是 v ，如果是对手是 -v.
        返回:
            v: 当前局面的状态
        """
        s = game.get_key()

        if self.depth>1000: return 0

        if game.terminal: self.Es[s] = -game.pieceheight 

        # 如果得分不等于0，标志探索结束
        if s in self.Es: return self.Es[s]

        # 如果当前状态没有子节点，增加子节点
        # 增加 Ps[s] Vs[s] Ns[s]
        if s not in self.Ps:                          
            # 获得当前局面的概率 和 局面的打分, 这个已经过滤掉了不可用走法
            act_probs, v = self._policy(game)

            probs = np.zeros(game.actions_num)
            for act, prob in act_probs:
                probs[act] = prob

            # alpha=1的时候，dir机会均等，>1 强调均值， <1 强调两端
            # 国际象棋 0.3 将棋 0.15 围棋 0.03
            # 取值一般倾向于 alpha = 10/n 所以俄罗斯方块取 2
            # dirichlet_alpha=2            
            # dirichlet_probs = np.random.dirichlet([dirichlet_alpha]*len(act_probs))
            # for (act, prob), noise in zip(act_probs, dirichlet_probs):
            #     probs[act] = 0.9*prob+0.1*noise

            self.Ps[s] = probs 

            self.Ns[s] = 0
            self.Vs[s] = v

            return v

        # 当前最佳概率和最佳动作
        cur_best = -float('inf')
        best_act = -1

        if best_act == -1:
            # 选择具有最高置信上限的动作
            for a in game.availables:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self._c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self._c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # 加一个EPS小量防止 Q = 0 

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        act = game.position_to_action(a)

        # prev_pieceheight = game.pieceheight
        # prev_EmptyCount = game.getEmptyCount()

        game.step(act)

        # 如果差两步，游戏结束
        # if game.state==1 and other_game.piececount>0:
        #     if game.pieceheight-other_game.pieceheight>1:
        #         game.terminal = True

        self.depth = self.depth +1

        # 现实奖励
        if game.state == 1 and game.piececount - game.piececount_mark >1:
            v = (game.getEmptyCount()+game.getHeightStd())/-10
        else:
            v = self.search(game)

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

    # c_puct MCTS child权重， 用来调节MCTS搜索深度，越大搜索越深，越相信概率，越小越相信Q 的程度 默认 5
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000):
        """初始化参数"""
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)

    def set_player_ind(self, p):
        """指定MCTS的playerid"""
        self.player = p
        self.mcts.lable = "AI(%s)"%p

    def reset_player(self):
        self.mcts.reset()

    def get_action(self, game, curr_player, temp=0):        
        """计算下一步走子action"""
        move_probs = np.zeros(game.actions_num)
        if not game.terminal:  # 如果游戏没有结束
            # 训练的时候 temp = 1
            # temp 导致 N^(1/temp) alphaezero 前 30 步设置为1 其余设置为无穷小即act_probs只取最大值
            # temp 越大导致更均匀的搜索

            # 对于俄罗斯方块，每个方块放下的第一个方块可以探索一下
            # if game.piececount>=1:
            #     temp = 0
            # temp = 1000

            acts, act_probs, act_qs, act_ps, state_v, state_n = self.mcts.get_action_probs(game, temp)
            depth = self.mcts.max_depth
            move_probs[acts] = act_probs
            max_probs_idx = np.argmax(act_probs)    
            max_qs_idx = np.argmax(act_qs) 
            max_ps_idx = np.argmax(act_ps)    

            if max_probs_idx == max_qs_idx:
                idx = max_probs_idx
            else:
                for i, qs in enumerate(act_qs):
                    if qs<act_qs[max_probs_idx]:
                        act_probs[i]=0
                act_probs = act_probs/np.sum(act_probs)        
                idx = np.random.choice(range(len(acts)), p=act_probs) 

            # if True or temp==0 or len(acts)==1 or game.piecesteps>2 :
            #     idx = max_probs_idx
            # else:
            # alphazero，默认p为0.75
            # p = 0.75
            # a=1的时候，dir机会均等，>1 强调均值， <1 强调两端
            # 国际象棋 0.3 将棋 0.15 围棋 0.03
            # 取值一般倾向于 a = 10/n 所以俄罗斯方块取 2
            # a = 2                  
            # dirichlet = np.random.dirichlet(a * np.ones(len(act_probs)))
            # idx = np.random.choice(range(len(acts)), p=p*act_probs + (1.0-p)*dirichlet)                                                                     
            # idx = np.random.choice(range(len(acts)), p=act_probs)                                                                     

            # if max_qs_idx!=max_probs_idx and random.random()<(act_qs[max_qs_idx]-act_qs[max_probs_idx])*act_probs[max_qs_idx]:
            #     idx = max_qs_idx

            action = acts[idx]
            qval = act_qs[max_probs_idx]

            if idx!=max_probs_idx:
                print("    random", game.position_to_action_name(acts[max_probs_idx]), "==>",  game.position_to_action_name(acts[idx]), \
                           "p:", act_probs[max_probs_idx], "==>", act_probs[idx], "q:", act_qs[max_probs_idx], "==>", act_qs[idx])  

            acc_ps = 1 if max_ps_idx==max_probs_idx else 0
            return action, move_probs, state_v, qval, acc_ps, depth, state_n
        else:
            print("WARNING: game is terminal")

    def __str__(self):
        return "AI {}".format(self.player)