from itertools import count
import logging
from math import sqrt
import copy
import random
import numpy as np
import time
from datetime import timedelta
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
        self.markscore = 0
        self.markfailtop = 0
        self.markEmptyCount = 0
        
    def step(self,act:int):
        return self.game.step(act)                      
        
    def terminal(self):
        return self.game.terminal
    
    # 2.049623722920656e-06  
    def __hash__(self):
        return self.game.get_key()

    def __eq__(self, other):
        return hash(self)==hash(other)

    def availables(self):
        return self.game.availables
    
    # 0.00022098371224809986  
    def clone(self):
        state = State(copy.deepcopy(self.game))
        state.search = self.search
        state.markscore = self.markscore
        state.markfailtop = self.markfailtop
        state.markEmptyCount = self.markEmptyCount  
        return state
    
    def mark(self):
        self.markscore=self.game.score
        self.markfailtop=self.game.failtop
        self.markEmptyCount=self.game.emptyCount
        
@njit(cache=True)
def selectAction(s:int, availables, _c_puct:float, Ps, Ns, Qsa, Nsa):
    # 如果有一次都没有探索的，返回
    # 开销 8.05366921094275e-05 S
    # njit 4.298482243524901e-05 S
    if np.min(Nsa[s][availables==1])==0:
        return np.argmax(Nsa[s]+availables == 1)
    q = Qsa[s]+ _c_puct * availables * Ps[s] * sqrt(Ns[s]) / Nsa[s]
    
    # 选择最大q的
    nz_idx = np.nonzero(availables)
    max_q_idx = nz_idx[0][np.argmax(q[nz_idx])]

    return max_q_idx
        
    # EPS = 1e-8
    # 开销 6.048132081767674e-05 S
    # njit 4.2826029136094145e-05 S
    # cur_best:float = -100000
    # best_act:int = -1
    # if best_act == -1:
    #     # 选择具有最高置信上限的动作   
    #     for a in range(len(availables)):     
    #         if availables[a]==0: continue                   
    #         if Qsa[s][a]!=0:
    #             u = Qsa[s][a] + _c_puct * Ps[s][a] * sqrt(Ns[s]) / Nsa[s][a]
    #         else:
    #             # 由于奖励都是正数，所以需要所有的步骤至少探索一次
    #             return a
    #             # u = _c_puct * Ps[s][a] * sqrt(Ns[s] + EPS)  # 加一个EPS小量防止 Q = 0                 
    #         if u > cur_best:
    #             cur_best = u
    #             best_act = a    
    # return best_act

@njit(cache=True)
# njit 9.572226293848707e-06
# 2.3297934257853874e-05
def updateQN(s:int, a:int, v:float, Ns, Qsa, Nsa, actions_num):
    Nsa[s][a] += 1
    # Qsa[s][a] += (v- Qsa[s][a])/Nsa[s][a]
    # Qsa[s][a] += (r+v- Qsa[s][a])/Nsa[s][a]   
    
    # DQN : q = r + 0.99*max(q_next)
    # Qsa[s][a] = (Qsa[s][a]*(Nsa[s][a]-1)+v)/Nsa[s][a]
    # b = (Qsa[s][a]*(Nsa[s][a]-1)+v -QSa[s][a]*Nsa[s][a])/Nsa[s][a]
    # b = (v-Qsa[s][a])/Nsa[s][a]
    
    # Qsa[s][a] += (r/Nsa[s][a]+v-Qsa[s][a])/Nsa[s][a]
    # Qsa[s][a] += (v- Qsa[s][a])/Nsa[s][a]
    delta =  (v - Qsa[s][a])/Nsa[s][a]
    Qsa[s][a] += delta
    
    # Qsa[s][a] = np.tanh(Qsa[s][a])
    Ns[s] += 1

@njit(cache=True)   
# njit  0.00022558832222352673
# 0.0001987227917925678
def expandPN(s:int, availables, act_probs, Ps, Ns, Nsa, Qsa, actions_num):
    # probs = np.zeros(actions_num, dtype=np.float32)
    # for i in range(len(availables)):
    #     if availables[i]==0: continue
    #     probs[i]=act_probs[i]
    probs = (act_probs*availables).astype(np.float32)
    Ps[s] = probs 
    Ns[s] = 0
    Nsa[s] = np.zeros(actions_num, dtype=np.int64)
    Qsa[s] = np.zeros(actions_num, dtype=np.float32)

@njit(cache=True)
def checkNeedExit(s:int, Nsa)->bool:
    max_v = np.max(Nsa[s])
    return max_v>0 and max_v/np.sum(Nsa[s])>0.8

# @njit(cache=True)
#njit 0.00561012750790443
# 0.0007797207943228788
def getprobsFromNsa(s:int, temp:float, availables, actions_num, Nsa):
    if temp == 0:
        probs = np.zeros(actions_num, dtype=np.float32)    
        probs[np.argmax(Nsa[s])] = 1        
    else:
        m_sum = np.sum(Nsa[s])
        if m_sum==0:
            probs = availables/np.sum(availables)
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
        value_type = numba.types.float32
    )
    
def getEmptySAF_Dict():
    return numba.typed.Dict.empty(
        key_type = numba.types.int64,
        value_type = numba.types.float32[:]
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
        # self.Es = getEmptySF_Dict()     # 保存游戏最终得分 key: s
        self.Vs = getEmptySF_Dict()     # 保存游戏局面差异奖励 key: s
        print("create mcts, c_puct: {}, n_playout: {}".format(c_puct, n_playout))
        self.t = 0
        self.c = 1
        self.start_time = time.time()
        self.limit_depth = 200
    
    def get_action_probs(self, state:State, temp:float=1):
        """
        获得mcts模拟后的最终概率， 输入游戏的当前状态 s
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """       
        s:int = hash(state)
        
        self.max_depth:int = 0
        self.simulation_count = 0
        state.mark()
        # for n in range(self._n_playout):
        self.limit_depth=20
        while True:
            self.simulation_count += 1
            
            # t = time.time()     
            state_:State =state.clone()
            # self.c += 1
            # self.t += time.time()-t 
            
            self.search(state_) 
            depth = state_.game.piececount-state.game.piececount
            if depth > self.max_depth: self.max_depth = depth
            if self.simulation_count>=10 and depth>self.limit_depth and state_.game.state==1 and self.Ns[s]>=100: break
            if self.simulation_count >= self._n_playout and state_.game.state==1 : break 

        probs = getprobsFromNsa(s, temp, state.availables(), state.actions_num, self.Nsa)                       
        
        qs = self.Qsa[s] 
        ps = self.Ps[s]         
        v:float = self.Vs[s]
        ns:float = self.Ns[s]
        max_p = np.argmax(ps)
        
        nz_idx = np.nonzero(state.availables())
        max_q_idx = nz_idx[0][np.argmax(qs[nz_idx])]
        
        # max_q = np.nanargmax(np.where(qs!=0, qs, np.nan))
        
        game = state.game
        if game.show_mcts_process or game.state == 1 :
            if game.state == 1: game.print()
            nz_idx = np.nonzero(state.availables())
            run_time = round(time.time()-self.start_time)
            print(timedelta(seconds=run_time), game.steps, game.fallpiece["shape"], \
                  "ns:", str(ns).rjust(4), "/", str(self.simulation_count).ljust(4), "depth:", str(self.max_depth).ljust(3), \
                #   "\tQ:", round(v,2), "-->",round(qs[max_p],2), '/', round(qs[max_q],2), \
                  "\tV:", round(v,2), "-->", round(qs[max_q_idx],2), \
                  "\t%s %s:"%(game.position_to_action_name(max_q_idx),game.position_to_action_name(max_p)), \
                  round(ps[max_p],2), "-->", round(probs[max_p],2), \
                  "\tQs:", qs, "var", np.var(qs[nz_idx]))
            # 如果这一局已经超过了20分钟
            # if run_time>20*60 and self.limit_depth!=20:
            #     print("limit max depth to 20")
            #     self.limit_depth=20
                
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
        # if s in self.Es: 
        #     return self.Es[s]

        # 如果当前状态没有子节点，增加子节点
        # 增加 Ps[s] Vs[s] Ns[s]
        if s not in self.Ps:                          
            # 获得当前局面的概率 和 局面的打分, 这个已经过滤掉了不可用走法
            act_probs, v = self._policy(state.game) 
              
            expandPN(s, state.availables(), act_probs, self.Ps, self.Ns, self.Nsa, self.Qsa, state.actions_num)             

            self.Vs[s] = v
            return v

        # 当前最佳概率和最佳动作
        # 比较 Qsa[s, a] + c_puct * Ps[s,a] * sqrt(Ns[s]) / Nsa[s, a], 选择最大的
        a = selectAction(s, state.availables(), self._c_puct, self.Ps, self.Ns, self.Qsa, self.Nsa)
        
        # _r = state.game.score
        # _c = state.game.emptyCount
        
        state.step(a)
        r = 0
        if state.game.state==1:
            r += (state.game.score-state.markscore) * state.game.exrewardRate
        if r < -2: r = -2
        
        # 如果游戏结束
        if state.terminal(): 
            # (state.game.state==1 and state.game.emptyCount-state.markEmptyCount>4) or \
            
            # self.Es[s] = -1
            # v = -1 + np.min(self.Qsa[s])
            v = -2
            # v = -1
            r = -2
        else:
            # 现实奖励
            # 按照DQN，  q[s,a] += 0.1*(r+ 0.99*(max(q[s+1])-q[s,a])
            # 目前Mcts， q[s,a] += v[s+1]/Nsa[s,a]
            v = self.search(state)
            # r = np.tanh(r)
        
        # r *= state.game.exrewardRate 
        # r = np.tanh(r)
        # if r>1: r = 1
        # if r<-1: r= -1
        # 更新 Q 值 和 访问次数
        # v = r + (v - 0.01)
        # v *= state.game.exrewardRate
        v += r
        updateQN(s, a, v, self.Ns, self.Qsa, self.Nsa, state.actions_num)

        # print(v, self.Qsa[s][a], v-self.Qsa[s][a])
        # v -= self.Qsa[s][a]
        
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
            # 动作数，概率，每个动作的Q，原始概率，当前局面的v，当前局面的总探索次数 
            act_probs, act_qs, act_ps, state_v, state_n = self.mcts.get_action_probs(state, temp)
            depth = self.mcts.max_depth
            
            # max_probs_idx = np.argmax(act_probs)    
            
            availables = state.availables()
            nz_idx = np.nonzero(availables)
            max_qs_idx = nz_idx[0][np.argmax(act_qs[nz_idx])]
            # var_qs = np.var(act_qs[nz_idx])
            # print(var_qs)
            max_ps_idx = nz_idx[0][np.argmax(act_ps[nz_idx])]

            # if var_qs>0.01:
            #     if game.is_replay:
            #         p = 0.75
            #         dirichlet = np.random.dirichlet(2 * np.ones(len(act_probs)))
            #         idx = np.random.choice(range(ACTONS_LEN), p=p*act_probs + (1.0-p)*dirichlet)                    
            #         # # idx = np.random.choice(range(ACTONS_LEN), p=act_probs)
            #         # idx = max_qs_idx
            #     else:
            #         idx = max_ps_idx
            # else:
            #     idx = np.random.choice(range(ACTONS_LEN), p=act_probs)    

            # 如果当前概率和推定概率一致,且都大于0.9,不需要随机
            if max_qs_idx==max_ps_idx and act_probs[max_qs_idx]>0.9 and act_ps[max_ps_idx]>0.9:
                idx = max_ps_idx
            else:
                p = 0.75                
                dirichlet = np.random.dirichlet(2 * np.ones(len(nz_idx[0])))
                dirichlet_probs = np.zeros_like(act_probs, dtype=np.float64)
                dirichlet_probs[nz_idx] = dirichlet
                idx = np.random.choice(range(ACTONS_LEN), p=p*act_probs + (1.0-p)*dirichlet_probs)

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

            if idx!=max_qs_idx:
                print("\t\trandom", game.position_to_action_name(max_qs_idx), "==>",  game.position_to_action_name(idx), \
                           "p:", round(act_probs[max_qs_idx],2), "==>", round(act_probs[idx],2), "q:", round(act_qs[max_qs_idx],2), "==>", round(act_qs[idx],2))  

            acc_ps = 0 if abs(act_ps[idx]-act_probs[idx])>0.4 else 1

            return action, qval, act_probs, state_v, max_qval, acc_ps, depth, state_n
        else:
            print("WARNING: game is terminal")

    def __str__(self):
        return "AI {}".format(self.player)