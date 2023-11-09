from itertools import count
import logging
from math import sqrt
import copy
import random
import numpy as np
import time
from agent_numba import ACTONS_LEN
from numba import njit
import numba

jit_args = {'nopython': True, 'cache': True, 'fastmath': True}

# 概率
class State():
    def __init__(self, game):
        self.game = game
        # 最大递归搜索深度
        self.search=0
        # 动作的类型数目
        self.actions_num = ACTONS_LEN
        self._availables = numba.typed.List.empty_list(numba.types.int64)
        
    def step(self,act:int):
        self.game.step(act)                      
        
    def terminal(self):
        return self.game.terminal
      
    def __hash__(self):
        return self.game.get_key()

    def __eq__(self, other):
        return hash(self)==hash(other)

    def availables(self):
        return self.game.availables

    def availables_nb(self):
        self._availables.clear()
        for act in self.game.availables:
            self._availables.append(act)
        return self._availables
        
    def clone(self):
        game = copy.deepcopy(self.game)
        state = State(game)
        return state

@njit
def selectAction(s:int, availables, _c_puct:float, Ps, Ns, Qsa, Nsa):
    # EPS = 1e-8
    cur_best:float = -100000
    best_act:int = -1
    if best_act == -1:
        # 选择具有最高置信上限的动作             
        for a in availables:            
            if Qsa[s][a]!=0:
                u = Qsa[s][a] + _c_puct * Ps[s][a] * sqrt(Ns[s]) / Nsa[s][a]
            else:
                # 由于奖励都是正数，所以需要所有的步骤至少探索一次
                return a
                # u = _c_puct * Ps[s][a] * sqrt(Ns[s] + EPS)  # 加一个EPS小量防止 Q = 0                 
            if u > cur_best:
                cur_best = u
                best_act = a    
    return best_act

@njit
def updateQN(s:int, a:int, v:float, Ns, Qsa, Nsa, actions_num):
    Nsa[s][a] += 1
    Qsa[s][a] += (v- Qsa[s][a])/Nsa[s][a]
    Ns[s] += 1

@njit    
def expandPN(s:int, availables, act_probs, Ps, Ns, Nsa, Qsa, actions_num):
    probs = np.zeros(actions_num, dtype=np.float64)
    for i in availables:
        probs[i]=act_probs[i]
    Ps[s] = probs 
    Ns[s] = 0
    Nsa[s] = np.zeros(actions_num, dtype=np.int64)
    Qsa[s] = np.zeros(actions_num, dtype=np.float64)

@njit
def checkNeedExit(s:int, Nsa)->bool:
    max_v = np.max(Nsa[s])
    return max_v>0 and max_v/np.sum(Nsa[s])>0.8

@njit
def getprobsFromNsa(s:int, temp:float, availables, actions_num, Nsa):
    probs = np.zeros(actions_num, dtype=np.float64)    
    if temp == 0:
        probs[np.argmax(Nsa[s])] = 1        
    else:
        m_sum = np.sum(Nsa[s])
        if m_sum==0:
            avg_v = 1/len(availables)
            for a in availables:
                probs[a] = avg_v        
        else:
            if temp == 1:
                probs = Nsa[s]/m_sum
            else:
                probs = np.power(Nsa[s],1/temp)
                probs = probs/np.sum(probs)                
    return probs

def getEmptySF_Dict():
    return numba.typed.Dict.empty(
        key_type = numba.types.int64,
        value_type = numba.types.float64
    )

def getEmptySAF_Dict():
    return numba.typed.Dict.empty(
        key_type = numba.types.int64,
        value_type = numba.types.float64[:]
    )

def getEmptySV_Dict():
    return numba.typed.Dict.empty(
        key_type = numba.types.int64,
        value_type = numba.types.int64
    )
    
def getEmptySAV_Dict():
    return numba.typed.Dict.empty(
        key_type = numba.types.int64,
        value_type = numba.types.int64[:]
    )

class MCTS():
    def __init__(self, policy_value_fn, c_puct:float=5, n_playout:int=10000):
        self._policy = policy_value_fn      # 概率估算函数
        self._c_puct:float = c_puct               # 参数
        self._n_playout:int = n_playout         # 做几次探索
        self.Qsa = getEmptySAF_Dict()  # 保存 Q 值, key: s,a
        self.Nsa = getEmptySAV_Dict()  # 保存 遍历次数 key: s,a
        self.Ns = getEmptySV_Dict()     # 保存 遍历次数 key: s
        self.Ps = getEmptySAF_Dict()     # 保存 动作概率 key: s, a
        self.Es = getEmptySF_Dict()     # 保存游戏最终得分 key: s
        self.Vs = getEmptySF_Dict()     # 保存游戏局面差异奖励 key: s
        self.Ss = getEmptySF_Dict()     # 保存游戏局面全局奖励 key: s        
        print("create mcts, c_puct: {}, n_playout: {}".format(c_puct, n_playout))
        self.ext_reward:bool = True # 是否中途额外奖励
    
    def get_action_probs(self, state:State, temp:float=1):
        """
        获得mcts模拟后的最终概率， 输入游戏的当前状态 s
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """       
        s:int = hash(state)
        
        self.max_depth:int = 0
        self.depth:int = 0
        self.simulation_count = 0
        self.available_acts = state.availables()
        
        for n in range(self._n_playout*2):
            self.depth = 0
            self.simulation_count = n+1
            
            state_:State =state.clone()
            
            self.search(state_) 
            
            if self.depth>self.max_depth: self.max_depth = self.depth
            if n >= self._n_playout//2-1 and state_.game.state==1 and checkNeedExit(s, self.Nsa): break

        probs = getprobsFromNsa(s, temp, state.availables_nb(), state.actions_num, self.Nsa)                       

        qs = self.Qsa[s] 
        ps = self.Ps[s]         
        v:float = self.Vs[s]
        ns:float = self.Ns[s]
        max_p = np.argmax(ps)
        
        game = state.game
        if game.show_mcts_process or game.state == 1 :
            if game.state == 1: game.print()
            act = np.argmax(probs)
            print(time.strftime('%m-%d %H:%M:%S',time.localtime(time.time())), game.steps, game.fallpiece["shape"], \
                  "ns:", str(ns).rjust(4), "/", str(self.simulation_count).ljust(4), "depth:", str(self.max_depth).ljust(3), \
                  "value:", round(v,2), "\t", act, self.Nsa[s], "Q:", round(np.max(qs),2), "P:", round(ps[max_p],2), "-->", round(probs[max_p],2))
        # 动作数，概率，每个动作的Q，原始概率，当前局面的v，当前局面的总探索次数
        return probs, qs, ps, v, ns

    def search(self, state:State):
        """
        蒙特卡洛树搜索        
        返回:
            v: 当前局面的状态
        """
        s = hash(state)
         
        # 如果得分不等于0，标志探索结束
        if s in self.Es: 
            return self.Es[s]

        # 如果当前状态没有子节点，增加子节点
        # 增加 Ps[s] Vs[s] Ns[s]
        if s not in self.Ps:                          
            # 获得当前局面的概率 和 局面的打分, 这个已经过滤掉了不可用走法
            act_probs, v = self._policy(state.game)   
            expandPN(s, state.availables_nb(), act_probs, self.Ps, self.Ns, self.Nsa, self.Qsa, state.actions_num)             
            self.Ss[s] = state.game.score
            self.Vs[s] = v
            return v

        # 当前最佳概率和最佳动作
        # 比较 Qsa[s, a] + c_puct * Ps[s,a] * sqrt(Ns[s]) / Nsa[s, a], 选择最大的
        a = selectAction(s, state.availables_nb(), self._c_puct, self.Ps, self.Ns, self.Qsa, self.Nsa)
 
        state.step(a)
        
        if state.terminal(): 
            self.Es[s] = -1
            v = -1
        else:
            # 现实奖励
            # 按照DQN，  q[s,a] += 0.1*(r+ 0.99*(max(q[s+1])-q[s,a])
            # 目前Mcts， q[s,a] += v[s]/Nsa[s,a]
            v = state.game.score-self.Ss[s] 
            v += self.search(state)
            self.depth += 1
            
        # 更新 Q 值 和 访问次数
        # q[s,a] += v[s]/Nsa[s,a]
        updateQN(s, a, v, self.Ns, self.Qsa, self.Nsa, state.actions_num)

        return v

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

    def get_action(self, game, curr_player, temp=0, avg_ns=0, avg_piececount=0):        
        """计算下一步走子action"""
        if not game.terminal:  # 如果游戏没有结束
            # 训练的时候 temp = 1
            # temp 导致 N^(1/temp) alphaezero 前 30 步设置为1 其余设置为无穷小即act_probs只取最大值
            # temp 越大导致更均匀的搜索
            state = State(game)
            # 清零game的方块下落次数
            game.downcount=0
            # 动作数，概率，每个动作的Q，原始概率，当前局面的v，当前局面的总探索次数 
            act_probs, act_qs, act_ps, state_v, state_n = self.mcts.get_action_probs(state, temp)
            depth = self.mcts.max_depth
            
            max_probs_idx = np.argmax(act_probs)    
            max_qs_idx = np.argmax(act_qs) 
            max_ps_idx = np.argmax(act_ps)

            if act_qs[max_qs_idx]>1:
                if max_qs_idx ==  max_ps_idx:
                    idx = max_qs_idx
                elif random.random()<0.8:
                    idx = max_ps_idx
                else:
                    idx = np.random.choice(range(ACTONS_LEN), p=act_probs)
            else:
                idx = max_ps_idx


            # if max_qs_idx ==  max_ps_idx:
            #     idx = max_qs_idx
            # elif random.random()>0.5:
            #     idx = max_ps_idx
            # else:
            #     for i, qs in enumerate(act_qs):
            #         if act_qs[max_probs_idx] - qs > 1:
            #             act_probs[i]=0
            #     act_probs = act_probs/np.sum(act_probs)        
            #     idx = np.random.choice(range(len(acts)), p=act_probs)                     

            action = idx
            qval = act_qs[idx]
            max_qval = np.max(act_qs)

            if idx!=max_probs_idx:
                print("\t\trandom", game.position_to_action_name(max_probs_idx), "==>",  game.position_to_action_name(idx), \
                           "p:", round(act_ps[max_probs_idx],2), "==>", round(act_ps[idx],2), "q:", round(act_qs[max_probs_idx],2), "==>", round(act_qs[idx],2))  

            acc_ps = 1 if max_ps_idx==max_probs_idx else 0

            return action, qval, act_probs, state_v, max_qval, acc_ps, depth, state_n
        else:
            print("WARNING: game is terminal")

    def __str__(self):
        return "AI {}".format(self.player)