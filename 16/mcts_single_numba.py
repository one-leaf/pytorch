from itertools import count
import logging
from math import sqrt
import copy
import random
import numpy as np
import time
from agent_numba import ACTONS_LEN
from typing import List, Dict, Tuple, Callable, Any
from numba import njit
import numba

jit_args = {'nopython': True, 'cache': True, 'fastmath': True}

# 概率
class State():
    def __init__(self, game:Any):
        self.game = game
        # 最大递归搜索深度
        self.search=0
        # 动作的类型数目
        self.actions_num = ACTONS_LEN
        self._availables = numba.typed.List.empty_list(numba.types.int64)
        
    def step(self,act:int):
        self.game.step(act)                      
        
    def terminal(self)->bool:
        return self.game.terminal
      
    def __hash__(self)->int:
        return self.game.get_key()

    def __eq__(self, other)->bool:
        return hash(self)==hash(other)

    def availables(self)->List[int]:
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

S_V = Dict[int, float]
S_P = Dict[int, List[float]]
SA_V = Dict[Tuple[int,int],float]
P_V_R = Tuple[List[Tuple[int,float]],float]

@njit
def selectAction(s:int, availables:List[int], _c_puct:float, Ps:S_V, Ns:S_V, Qsa:SA_V, Nsa:SA_V):
    EPS = 1e-8
    cur_best:float = -100000
    best_act:int = -1
    if best_act == -1:
        # 选择具有最高置信上限的动作             
        for a in availables:            
            if (s, a) in Qsa:
                u = Qsa[(s, a)] + _c_puct * Ps[s][a] * sqrt(Ns[s]) / Nsa[(s, a)]
            else:
                # 由于奖励都是正数，所以需要所有的步骤至少探索一次
                return a
                # u = _c_puct * Ps[s][a] * sqrt(Ns[s] + EPS)  # 加一个EPS小量防止 Q = 0                 
            if u > cur_best:
                cur_best = u
                best_act = a    
    return best_act

@njit
def updateQN(s:int, a:int, v:float, Ns:S_V, Qsa:SA_V, Nsa:SA_V):
    if (s, a) in Qsa:
        Nsa[(s, a)] += 1
        Qsa[(s, a)] += (v- Qsa[(s, a)])/Nsa[(s, a)]
    else:
        Nsa[(s, a)] = 1
        Qsa[(s, a)] = v
    Ns[s] += 1

@njit    
def expandPN(s:int, actions_num:int, availables:List[int], act_probs, v, Ps, Ns):
    probs = np.zeros(actions_num)    
    for i in availables:
        probs[i]=act_probs[i]
    Ps[s] = probs 
    Ns[s] = 0

@njit
def checkNeedExit(s:int, availables, Nsa)->bool:
    _act_visits = []
    for a in availables:
        if (s,a) in Nsa:
            _act_visits.append(Nsa[(s,a)])
    if len(_act_visits)==0: return False
    return max(_act_visits)/sum(_act_visits)>0.8

@njit
def getprobsFromNsa(s:int, temp:float, availables, Nsa):
    len_availables = len(availables)
    visits = np.zeros(len_availables)
    for i, a in enumerate(availables):
        if (s,a) in Nsa:
            visits[i]=Nsa[(s,a)]
    if temp == 0:
        bestA = np.argmax(visits)
        visits[bestA] = 1
    else:
        m_sum = np.sum(visits)
        if m_sum<=0:
            visits = np.ones(len_availables)/len_availables
        else:
            visits = np.power(visits,1/temp)/m_sum
    return visits


def getEmptySV_Dict():
    return numba.typed.Dict.empty(
        key_type = numba.types.int64,
        value_type = numba.types.float64
    )
    
def getEmptySAV_Dict():
    return numba.typed.Dict.empty(
        key_type = numba.types.UniTuple(numba.types.int64,2),
        value_type = numba.types.float64
    )
    
def getEmptySP_Dict():
    return numba.typed.Dict.empty(
        key_type = numba.types.int64,
        value_type = numba.types.float64[:]
    )


class MCTS():
    def __init__(self, policy_value_fn:Callable[[Any],P_V_R], c_puct:float=5, n_playout:int=10000):
        self._policy:Callable[[Any],P_V_R] = policy_value_fn      # 概率估算函数
        self._c_puct:float = c_puct               # 参数
        self._n_playout:int = n_playout         # 做几次探索
        self.Qsa:SA_V = getEmptySAV_Dict()  # 保存 Q 值, key: s,a
        self.Nsa:SA_V = getEmptySAV_Dict()  # 保存 遍历次数 key: s,a
        self.Ns:S_V = getEmptySV_Dict()     # 保存 遍历次数 key: s
        self.Ps:S_P = getEmptySP_Dict()     # 保存 动作概率 key: s, a
        self.Es:S_V = getEmptySV_Dict()     # 保存游戏最终得分 key: s
        self.Vs:S_V = getEmptySV_Dict()     # 保存游戏局面差异奖励 key: s
        self.Ss:S_V = getEmptySV_Dict()     # 保存游戏局面全局奖励 key: s        
        print("create mcts, c_puct: {}, n_playout: {}".format(c_puct, n_playout))
        self.ext_reward:bool = True # 是否中途额外奖励
    
    def get_action_probs(self, state:State, temp:float=1)->Tuple[List[int],List[float],List[float],float,int]:
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
        
        for n in range(self._n_playout):
            self.depth = 0
            self.simulation_count = n+1
            
            state_:State =state.clone()
            
            self.search(state_) 
            
            if self.depth>self.max_depth: self.max_depth = self.depth
            # if n >= self._n_playout//2-1 and state_.game.state==1 and checkNeedExit(s, state_.availables_nb(), self.Nsa): break

        probs = getprobsFromNsa(s, temp, state.availables_nb(), self.Nsa)                       

        act_Qs:List[Tuple[int,float]] = [(a, self.Qsa[(s, a)]) if (s, a) in self.Qsa else (a, 0) for a in self.available_acts]
        acts:List[float] = [a for a in self.available_acts]
        qs:List[float] = [av[1] for av in act_Qs]
        ps:List[float] = [self.Ps[s][a] if s in self.Ps else 0 for a in acts]
        v:float = 0 if s not in self.Vs else self.Vs[s]
        ns:float = 1 if s not in self.Ns else self.Ns[s]
        
        game = state.game
        if game.show_mcts_process or game.state == 1 :
            if game.state == 1: game.print()
            act = acts[np.argmax(probs)]
            print(time.strftime('%m-%d %H:%M:%S',time.localtime(time.time())), game.steps, game.fallpiece["shape"], \
                  "temp:", round(temp,2), "ns:", ns, "/", self.simulation_count, "depth:", self.max_depth, \
                  "value:", round(v,2), act, act_Qs)
        # 动作数，概率，每个动作的Q，原始概率，当前局面的v，当前局面的总探索次数
        return acts, probs, qs, ps, v, ns

    def search(self, state:State)->float:
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
            expandPN(s, state.actions_num, state.availables_nb(), act_probs, v, self.Ps, self.Ns)             
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
        updateQN(s, a, v, self.Ns, self.Qsa, self.Nsa)

        return v

    # 返回某个局面的action信息
    def getInfo(self, state:str, act:int)->Tuple[int,float,float]:
        s = state
        n = q = p = 0
        if (s, act) in self.Nsa: n = self.Nsa[(s, act)]
        if (s, act) in self.Qsa: q = self.Qsa[(s, act)]
        if s in self.Ps: p = self.Ps[s][act]
        return n, q, p

class MCTSPlayer(object):
    """基于模型指导概率的MCTS + AI player"""

    # c_puct MCTS child权重， 用来调节MCTS搜索深度，越大搜索越深，越相信概率，越小越相信Q 的程度 默认 5
    def __init__(self, policy_value_function:Callable[[Any],P_V_R], c_puct=5, n_playout=2000):
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
            # 动作数，概率，每个动作的Q，原始概率，当前局面的v，当前局面的总探索次数 
            acts, act_probs, act_qs, act_ps, state_v, state_n = self.mcts.get_action_probs(state, temp)
            depth = self.mcts.max_depth
            
            max_probs_idx = np.argmax(act_probs)    
            max_qs_idx = np.argmax(act_qs) 
            max_ps_idx = np.argmax(act_ps)

            if act_qs[max_qs_idx]>1:
                if max_qs_idx ==  max_ps_idx:
                    idx = max_qs_idx
                elif random.random()>0.5:
                    idx = max_ps_idx
                else:
                    idx = np.random.choice(range(len(acts)), p=act_probs)
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

            action = acts[idx]
            qval = act_qs[idx]
            max_qval = np.max(act_qs)

            if idx!=max_probs_idx:
                print("\t\trandom", game.position_to_action_name(acts[max_probs_idx]), "==>",  game.position_to_action_name(acts[idx]), \
                           "p:", act_ps[max_probs_idx], "==>", act_ps[idx], "q:", act_qs[max_probs_idx], "==>", act_qs[idx])  

            acc_ps = 1 if max_ps_idx==max_probs_idx else 0

            move_probs = np.zeros(ACTONS_LEN)
            move_probs[acts] = act_probs

            return action, qval, move_probs, state_v, max_qval, acc_ps, depth, state_n
        else:
            print("WARNING: game is terminal")

    def __str__(self):
        return "AI {}".format(self.player)