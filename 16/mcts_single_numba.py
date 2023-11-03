from itertools import count
import logging
from math import sqrt
import copy
import random
import numpy as np
import time
from agent import ACTIONS
from typing import Set, List, Dict, Tuple, Callable, Union, Sequence, Any
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
        self.actions_num = game.actions_num
        self.marks = {}
        self._availables = numba.typed.List.empty_list(numba.types.int64)
        
    def step(self,act:int):
        self.game.step(act)
        
    def terminal(self)->bool:
        return self.game.terminal
    
    # 中途奖励
    def reward(self)->float:
        v:float = 0
        if self.game.state == 1:
            if self.game.emptyCount>self.marks["emptyCount"]:
                v = self.marks["emptyCount"] - self.game.emptyCount
            elif self.game.exreward:                 
                v = (self.game.reward) + \
                    ((self.marks["emptyCount"] - self.game.emptyCount) + \
                    (self.marks["failtop"] - self.game.failtop) + \
                    (self.marks["heightDiff"] - self.game.heightDiff)*0.1)*self.game.exrewardRate        
            return v
        return v
    
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
        
    # 中途记录状态
    def mark(self):
        self.marks["score"] = self.game.score
        self.marks["piececount"] = self.game.piececount
        self.marks["emptyCount"] = self.game.emptyCount
        self.marks["failtop"] = self.game.failtop
        self.marks["heightDiff"] = self.game.heightDiff

    def clone(self):
        game = copy.deepcopy(self.game)
        state = State(game)
        state.marks=self.marks
        return state

S_V = Dict[int, float]
S_P = Dict[int, List[float]]
SA_V = Dict[Tuple[int,int],float]
P_V_R = Tuple[List[Tuple[int,float]],float]

@njit
def getBestAction(s:int, availables:List[int], _c_puct:float, Ps:S_V, Ns:S_V, Qsa:SA_V, Nsa:SA_V):
    EPS = 1e-8
    cur_best:float = -10000
    best_act:int = -1
    if best_act == -1:
        # 选择具有最高置信上限的动作             
        for a in availables:   
            if (s, a) in Qsa:
                u = Qsa[(s, a)] + _c_puct * Ps[s][a] * sqrt(Ns[s]) / (1 + Nsa[(s, a)])
            else:
                u = _c_puct * Ps[s][a] * sqrt(Ns[s] + EPS)  # 加一个EPS小量防止 Q = 0                 
            if u > cur_best:
                cur_best = u
                best_act = a    
    return best_act

@njit
def updateQN(s:int, a:int, v:float, Ns:S_V, Qsa:SA_V, Nsa:SA_V):
    if (s, a) in Qsa:
        Qsa[(s, a)] = (Nsa[(s, a)] * Qsa[(s, a)] + v) / (Nsa[(s, a)] + 1)
        Nsa[(s, a)] += 1
    else:
        Qsa[(s, a)] = v
        Nsa[(s, a)] = 1
    Ns[s] += 1

@njit    
def expandPN(s:int, actions_num:int, availables:List[int], act_probs, v, Ps, Ns, Vs):
    probs = np.zeros(actions_num)    
    for i in availables:
        probs[i]=act_probs[i]
    Ps[s] = probs 
    Ns[s] = 0
    Vs[s] = v

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
        self.Ns:S_V = getEmptySV_Dict()  # 保存 遍历次数 key: s
        self.Ps:S_P = getEmptySP_Dict()  # 保存 动作概率 key: s, a
        self.Es:S_V = getEmptySV_Dict()  # 保存游戏最终得分 key: s
        self.Vs:S_V = getEmptySV_Dict()  # 保存游戏局面打分 key: s # 这个不需要，只是缓存
        print("create mcts, c_puct: {}, n_playout: {}".format(c_puct, n_playout))
        self.ext_reward:bool = True # 是否中途额外奖励

    def reset(self):
        self.Qsa:SA_V = {}  # 保存 Q 值, key: s,a
        self.Nsa:SA_V = {}  # 保存 遍历次数 key: s,a
        self.Ns:S_V = {}  # 保存 遍历次数 key: s
        self.Ps:S_P = {}  # 保存 动作概率 key: s, a
        self.Es:S_V = {}  # 保存游戏最终得分 key: s
        self.Vs:S_V = {}  # 保存游戏局面打分 key: s # 这个不需要，只是缓存
    
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
        state.mark()
        self.available_acts = state.availables()
        
        for n in range(self._n_playout*2):
            self.depth = 0
            self.simulation_count = n+1
            
            state_:State =state.clone()
            
            self.search(state_) 
            
            if self.depth>self.max_depth: self.max_depth = self.depth
            if n >= self._n_playout/2-1 and (state_.game.state == 1 or state_.terminal) and checkNeedExit(s, state_.availables_nb(), self.Nsa): break

        probs = getprobsFromNsa(s, temp, state.availables_nb(), self.Nsa)
                        
        # act_visits:List[Tuple[int,float]] = [(a, self.Nsa[(s, a)]) if (s, a) in self.Nsa else (a, 0) for a in self.available_acts]
        # visits:List[float] = [av[1] for av in act_visits]

        act_Qs:List[Tuple[int,float]] = [(a, self.Qsa[(s, a)]) if (s, a) in self.Qsa else (a, 0) for a in self.available_acts]
        acts:List[float] = [a for a in self.available_acts]
        qs:List[float] = [av[1] for av in act_Qs]
        ps:List[float] = [self.Ps[s][a] if s in self.Ps else 0 for a in acts]
        v:float = 0 if s not in self.Vs else self.Vs[s]
        ns:float = 1 if s not in self.Ns else self.Ns[s]

        # if temp == 0:
        #     _probs:np.ndarray = np.zeros(len(visits))
        #     bestA:int = np.argmax(visits).tolist()
        #     _probs[bestA] = 1
        # else:
        #     m:np.ndarray = np.power(np.array(visits), 1/temp)
        #     m_sum:Any = np.sum(m)
        #     if m_sum<=0:
        #         v_len:int = len(visits)
        #         _probs:np.ndarray = np.ones(v_len)/v_len
        #     else:
        #         _probs:np.ndarray = m/m_sum
        # probs:List[float] = _probs.tolist()
        
        game = state.game
        if game.show_mcts_process or game.state == 1 :
            if game.state == 1: game.print()
            # info=[]
            # visits_sum:float=sum(visits)
            # if visits_sum==0: visits_sum=1
            # for idx in sorted(range(len(visits)), key=visits.__getitem__)[::-1]:
            #     act,visit = act_visits[idx]
            #     q:float = 0
            #     p:float = 0
            #     if (s, act) in self.Qsa: q:float = self.Qsa[(s, act)]
            #     if s in self.Ps: p:float = self.Ps[s][act]
            #     info.append([game.position_to_action_name(act), round(q,2), round(p,2),'>', round(visit/visits_sum,2),])  
            print(time.strftime('%m-%d %H:%M:%S',time.localtime(time.time())), game.steps, game.fallpiece["shape"], \
                  "temp:", round(temp,2), "ns:", ns, "/", self.simulation_count, "depth:", self.max_depth, \
                  "value:", round(v,2))

        return acts, probs, qs, ps, v, ns

    # @njit(parallel=True)
    def search(self, state:State)->float:
        """
        蒙特卡洛树搜索        
        NOTE: 返回当前局面的状态 [-1,1] 如果是当前玩家是 v ，如果是对手是 -v.
        返回:
            v: 当前局面的状态
        """
        s = hash(state)

        if state.terminal(): self.Es[s] = -2
         
        # 如果得分不等于0，标志探索结束
        if s in self.Es: return self.Es[s]

        # 如果当前状态没有子节点，增加子节点
        # 增加 Ps[s] Vs[s] Ns[s]
        if s not in self.Ps:                          
            # 获得当前局面的概率 和 局面的打分, 这个已经过滤掉了不可用走法
            act_probs, v = self._policy(state.game)               
            expandPN(s, state.actions_num, state.availables_nb(), act_probs, v, self.Ps, self.Ns, self.Vs)             
            return v

        # 当前最佳概率和最佳动作
        a = getBestAction(s, state.availables_nb(), self._c_puct, self.Ps, self.Ns, self.Qsa, self.Nsa)
 
        state.step(a)
        
        self.depth += 1

        # 现实奖励
        v = state.reward() + self.search(state)

        # 更新 Q 值 和 访问次数
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
        move_probs = np.zeros(game.actions_num)
        if not game.terminal:  # 如果游戏没有结束
            # 训练的时候 temp = 1
            # temp 导致 N^(1/temp) alphaezero 前 30 步设置为1 其余设置为无穷小即act_probs只取最大值
            # temp 越大导致更均匀的搜索

            # 对于俄罗斯方块，每个方块放下的第一个方块可以探索一下
            # if game.piececount>=1:
            #     temp = 0
            # temp = 1000
            state = State(game)
            acts, act_probs, act_qs, act_ps, state_v, state_n = self.mcts.get_action_probs(state, temp)
            depth = self.mcts.max_depth
            move_probs[acts] = act_probs
            max_probs_idx = np.argmax(act_probs)    
            max_qs_idx = np.argmax(act_qs) 
            max_ps_idx = np.argmax(act_ps)

            # idx = max_qs_idx
            # 直接用概率最大的走法
            if max_qs_idx ==  max_ps_idx:
                idx = max_ps_idx
            elif random.random()>0.8:
                idx = max_qs_idx
            else:
                idx = max_ps_idx 

            # 都兼顾
            # if max_qs_idx ==  max_ps_idx:
            #     idx = max_qs_idx
            # elif random.random()>0.25:
            #     idx = max_ps_idx
            # else:
            #     for i, qs in enumerate(act_qs):
            #         if act_qs[max_probs_idx] - qs > 0 and qs <= 0:
            #             act_probs[i]=0
            #     act_probs = act_probs/np.sum(act_probs)        
            #     idx = np.random.choice(range(len(acts)), p=act_probs) 


            #       
            # if random.random()>0.5**game.piececount:
            #     idx = max_ps_idx
            # else:
            #     p = 0.75
            #     a = 2
            #     dirichlet = np.random.dirichlet(a * np.ones(len(acts)))
            #     rp = p*np.array(act_ps) + (1.0-p)*dirichlet
            #     idx = np.random.choice(range(len(acts)), p=rp/np.sum(rp))

            # 尝试其他的走法
            # if max_qs_idx ==  max_ps_idx:
            #     idx = max_qs_idx                
            # elif (random.random() > 2*game.piececount/(avg_piececount+1))  or random.random()<(game.piececount - avg_piececount)/(avg_piececount+1):
            # if False or game.is_status_optimal():
            #     p = 0.75
            #     # a=1的时候，dir机会均等，>1 强调均值， <1 强调两端
            #     # 国际象棋 0.3 将棋 0.15 围棋 0.03
            #     # 取值一般倾向于 a = 10/n 所以俄罗斯方块取 2
            #     a = 2                  
            #     dirichlet = np.random.dirichlet(a * np.ones(len(act_probs)))
            #     idx = np.random.choice(range(len(acts)), p=p*act_probs + (1.0-p)*dirichlet)
            # # 20% 按得分大于当前的概率
            # else:
            #     for i, qs in enumerate(act_qs):
            #         if act_qs[max_probs_idx] - qs > 0 and qs <= 0:
            #             act_probs[i]=0
            #     act_probs = act_probs/np.sum(act_probs)        
            #     idx = np.random.choice(range(len(acts)), p=act_probs) 

            action = acts[idx]
            qval = np.max(act_qs)

            if idx!=max_probs_idx:
                print("\t\trandom", game.position_to_action_name(acts[max_probs_idx]), "==>",  game.position_to_action_name(acts[idx]), \
                           "p:", act_ps[max_probs_idx], "==>", act_ps[idx], "q:", act_qs[max_probs_idx], "==>", act_qs[idx])  

            acc_ps = 1 if max_ps_idx==max_probs_idx else 0

            return action, move_probs, state_v, qval, acc_ps, depth, state_n
        else:
            print("WARNING: game is terminal")

    def __str__(self):
        return "AI {}".format(self.player)