from itertools import count
import logging
from math import sqrt
import copy
import random
import numpy as np
import time
from datetime import timedelta
from agent_numba import ACTONS_LEN
# from numba import njit
# import numba
import torch

jit_args = {'nopython': True, 'cache': True, 'fastmath': True}

# 概率
class State():
    def __init__(self, game):
        self.game = game
        # 动作的类型数目
        self.actions_num = ACTONS_LEN
        self.markscore = 0
        self.markfailtop = 0
        self.markEmptyCount = 0  
        self.markPiececount = 0
        self.markSteps = 0
        
    def __hash__(self):
        return self.game.key

    def __eq__(self, other):
        return hash(self)==hash(other)

    def availables(self):
        return self.game.availables
    
    # 0.00022098371224809986  
    def clone(self):
        # state = State(copy.deepcopy(self.game))
        state = State(self.game.clone())
        return state   
        
#@njit(cache=True)
def selectAction(s:int, availables, _c_puct:float, Ps, Ns, Qsa, Nsa):
    # 如果有一次都没有探索的，返回
    # 开销 8.05366921094275e-05 S
    # njit 4.298482243524901e-05 S
    
    q = Qsa[s]+ _c_puct * Ps[s] * sqrt(Ns[s]) / (Nsa[s]+1)
    
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

#@njit(cache=True)
# njit 9.572226293848707e-06
# 2.3297934257853874e-05
def updateQN(s:int, a:int, v:float, Ns, Qsa, Nsa, actions_num):
    Nsa[s][a] += 1
    Ns[s] += 1
    # Qsa[s][a] += (v- Qsa[s][a])/Nsa[s][a]
    # Qsa[s][a] += (r+v- Qsa[s][a])/Nsa[s][a]   
    
    # DQN : q = r + 0.999*max(q_next)
    # Qsa[s][a] = (Qsa[s][a]*(Nsa[s][a]-1)+v)/Nsa[s][a]
    # b = (Qsa[s][a]*(Nsa[s][a]-1)+v -QSa[s][a]*Nsa[s][a])/Nsa[s][a]
    # b = (v-Qsa[s][a])/Nsa[s][a]
    
    # Qsa[s][a] = r[s][a] + gamma*(V[s+1])
    # V[s] = sum(Qsa[s][a]*Nsa[s][a])/Ns[s]
    # Ns[s] = Ns[s] + 1
    
    # Q[t+1] = Q[t] - a(Q[t] - v[t])   a=1/t     
    Qsa[s][a] += (v - Qsa[s][a])/Nsa[s][a]  # 棋类博弈使用
    # Qsa[s][a] += (v - Qsa[s][a])/sqrt(Nsa[s][a])
    
    # Qsa[s][a] = np.tanh(Qsa[s][a])

#@njit(cache=True)   
# njit  0.00022558832222352673
# 0.0001987227917925678
def expandPN(s:int, availables, act_probs, v, Ps, Ns, Nsa, Qsa, actions_num):
    _p = np.exp((act_probs-np.max(act_probs)))
    
    _p[availables==0]=0
    _p_sum = np.sum(_p)
    if _p_sum > 0:     
        # if v<0 and np.max(_p/_p_sum)>0.99:             
        #     _p[availables==1] += 0.002 
        #     _p_sum = np.sum(_p)
        # if np.max(_p/_p_sum)>0.95:        
        #     probs = availables/np.sum(availables)
        #     Ps[s] = probs*0.05 + _p*0.95/_p_sum
        # else:
        Ps[s] = _p/_p_sum
    else:
        Ps[s] = availables/np.sum(availables) 
    Ns[s] = 0
    Nsa[s] = np.zeros(actions_num, dtype=np.int64)
    Qsa[s] = np.zeros(actions_num, dtype=np.float64)

#@njit(cache=True)
def checkNeedExit(s:int, Nsa)->bool:
    max_v = np.max(Nsa[s])
    return max_v>0 and max_v/np.sum(Nsa[s])>0.8

# @njit(cache=True)
#njit 0.00561012750790443
# 0.0007797207943228788
def getprobsFromNsa(s:int, temp:float, availables, actions_num, Nsa):
    if temp == 0:
        probs = np.zeros(actions_num, dtype=np.float64)    
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
    return {}
    # return numba.typed.Dict.empty(
    #     key_type = numba.types.int64,
    #     value_type = numba.types.float64
    # )
    
def getEmptySAF_Dict():
    return {}
    # return numba.typed.Dict.empty(
    #     key_type = numba.types.int64,
    #     value_type = numba.types.float64[:]
    # )

def getEmptySV_Dict():
    return {}
    # return numba.typed.Dict.empty(
    #     key_type = numba.types.int64,
    #     value_type = numba.types.int64
    # )

def getEmptySAV_Dict():
    return {}
    # return numba.typed.Dict.empty(
    #     key_type = numba.types.int64,
    #     value_type = numba.types.int64[:]
    # )

class MCTS():
    def __init__(self, policy_value_fn, c_puct:float=5, n_playout:int=10000, limit_depth=20):
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
        self.limit_depth = limit_depth
        self.reward_piececount = 2      # 放置几个方块数后奖励一次
        # self.extra_reward = False
        
    
    def get_action_probs(self, state:State, temp:float=1):
        """
        获得mcts模拟后的最终概率， 输入游戏的当前状态 s
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """     
        s = hash(state)        
        self.max_depth:int = (0,0)
        self.simulation_count = 0
        die_count = 0

        state_ = None

        find_end = False
        # self._n_playout = 256    # 最少256次，概率才会比较准确
        for n in range(self._n_playout):
            self.simulation_count += 1

            state_:State = state.clone()
            self.search(state_) 
            
            depth = state_.game.piececount-state.game.piececount
            step_depth = state_.game.steps-state.game.steps
            if state_.game.terminal:
                die_count += 1 
                find_end = True
                
            self.max_depth = (depth, step_depth)

            if self._n_playout>100 and self.Ns[s]>=self._n_playout*2 :
                break

        self._policy(state_.game, only_Cache_Next=True) 

        probs = getprobsFromNsa(s, temp, state.availables(), state.actions_num, self.Nsa)                       
        
        qs = self.Qsa[s] 
        ps = self.Ps[s]     
        ns = self.Nsa[s]#/(self.Ns[s]+1)
        v:float = self.Vs[s] 
        nsv:float = self.Ns[s]
        max_p = np.argmax(ps)
        
        nz_idx = np.nonzero(state.availables())
        max_q_idx = nz_idx[0][np.argmax(qs[nz_idx])]
        max_n_idx = nz_idx[0][np.argmax(probs[nz_idx])]
        
        mask = "-" if max_n_idx==max_q_idx else "?"
                
        game = state.game
        if game.show_mcts_process or game.state == 1 :
            nz_idx = np.nonzero(state.availables())
            run_time = round(time.time()-game.start_time)
            print(timedelta(seconds=run_time), game.steps, game.fallpiece["shape"], \
                  "ns:", str(nsv).rjust(4), "/", str(self.simulation_count).ljust(4), "depth:", str(self.max_depth).ljust(3), \
                  "end:", die_count, "\tv:", round(v,2), "-->", round(qs[max_p],2), "\t", \
                  mask, game.position_to_action_name(max_p), \
                  "Qs:", qs, "\tNs:", ns, "\tPs:", ps)
                
        # 动作数，概率，每个动作的Q，原始概率，当前局面的v，当前局面的总探索次数
        return probs, qs, ps, v, find_end

    def search(self, state:State):
        """
        蒙特卡洛树搜索        
        返回:
            v: 当前局面的状态
        """  
        if state.game.terminal: 
            return -2 #-1-1/(1+state.game.removedlines)  # 游戏结束，返回-1，奖励越多，分数越高   
        s = hash(state)

        # 如果当前状态没有子节点，增加子节点
        # 增加 Ps[s] Vs[s] Ns[s]
        availables = state.availables()
        
        if s not in self.Ps:                          
            # 获得当前局面的概率 和 局面的打分
            act_probs, v = self._policy(state.game) 
            expandPN(s, availables, act_probs, v, self.Ps, self.Ns, self.Nsa, self.Qsa, state.actions_num)             
            self.Vs[s] = v
            v = float(v)
            return v
            
        # 当前最佳概率和最佳动作
        # 比较 Qsa[s, a] + c_puct * Ps[s,a] * sqrt(Ns[s]) / Nsa[s, a], 选择最大的
        a = selectAction(s, availables, self._c_puct, self.Ps, self.Ns, self.Qsa, self.Nsa)
        
        _emptyCount = state.game.emptyCount
        _state, _removedlines = state.game.step(a)
        _emptyCount = state.game.emptyCount - _emptyCount
                
        v = self.search(state) 

        v += _removedlines*0.1   # 每消除一行奖励0.1，奖励越多，分数越高        
        v -= _emptyCount*0.01       # 每多一个空洞扣0.01
        
        # 外部奖励，放的方块越多越好
        # if state.game.state==1:
        #     v += 0.01
        # if v>2: v=2    
        # if v<-2: v=-2    
        updateQN(s, a, v, self.Ns, self.Qsa, self.Nsa, state.actions_num)
        
        return v

class MCTSPlayer(object):
    """基于模型指导概率的MCTS + AI player"""

    # c_puct MCTS child权重， 用来调节MCTS搜索深度，越大搜索越深，越相信概率，越小越相信Q 的程度 默认 5
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, limit_count=512, min_score=20, need_max_ps=False, need_max_ns=False):
        """初始化参数"""
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self.need_max_ps = need_max_ps
        self.need_max_ns = need_max_ns
        self.limit_count = limit_count
        self.n_playout = n_playout
        self.player = -1
        self.min_score = min_score
        self.cache = {}
        self.start_time = time.time()
        print("create mcts player, c_puct: {}, n_playout: {}, min_score: {}".format(c_puct, n_playout, min_score))
        
    def set_player_id(self, p):
        """指定MCTS的playerid"""
        self.player = p
        self.mcts.lable = "AI(%s)"%p

    def reset_player(self):
        self.mcts.reset()

    def get_action(self, game, temp=0):        
        """计算下一步走子action"""
        if not game.terminal:  # 如果游戏没有结束
            # 训练的时候 temp = 1
            # temp 导致 N^(1/temp) alphaezero 前 30 步设置为1 其余设置为无穷小即act_probs只取最大值
            # temp 越大导致更均匀的搜索
            
            # 为了防止无休止运行，runtime超过了30分钟，采用随机75%选择
            # if  game.removedlines <= self.min_score:
            #     self.start_time = time.time() # 重新开始计时

            # has_run_time=time.time()-self.start_time

            state = State(game)
            # 动作数概率，每个动作的Q，原始概率，当前局面的v，当前局面的总探索次数 
            act_probs, act_qs, act_ps, state_v, find_end = self.mcts.get_action_probs(state, temp)
            depth = self.mcts.max_depth[0]
            
            # max_probs_idx = np.argmax(act_probs)    
            
            availables = state.availables()

            if  hash(state) in self.cache :
                action = self.cache[hash(state)]
                if action < len(availables)-1:
                    availables[action]=0       
                
            nz_idx = np.nonzero(availables)[0]  # [0,2,3,4]
            
            # Qs
            max_qs_idx = nz_idx[np.argmax(act_qs[nz_idx])]
            
            # NS
            max_ns_idx = nz_idx[np.argmax(act_probs[nz_idx])]
            
            # PS            
            max_ps_idx = nz_idx[np.argmax(act_ps[nz_idx])]

            # print("       max_qs_idx:", max_qs_idx, "max_ns_idx:", max_ns_idx, "max_ps_idx:", max_ps_idx, "act_probs:", act_probs, "act_ps:", act_ps, "act_qs:", act_qs )
            idx = -1           
            
            # if self.need_max_ns:
            #     # idx = max_ns_idx
            #     # if has_run_time < 3600:
            #     #     idx = np.random.choice(range(ACTONS_LEN), p=act_probs)
            #     # else:
            #     idx = np.random.choice(range(ACTONS_LEN), p=act_probs)
            # elif self.need_max_ps:
            #     idx = max_ps_idx                          
            #     # idx = np.random.choice(range(ACTONS_LEN), p=act_ps)           
            
            # 如果Down+None的概率和为1， 则选择Down
            # if sum(act_probs[0:3])<0.01:
            #     idx = 4
            #     act_probs[3]=0.01
            #     act_probs[4]=1-sum(act_probs[0:4])
            
            # 如果有左右来回移动的动作   
            # if ("R" in game.piece_actions and idx == 1) or ("L" in game.piece_actions and idx == 2):
            #     avg_ps = (act_probs[1]+act_probs[2]+act_probs[4])/3
            #     act_probs[1] = avg_ps
            #     act_probs[2] = avg_ps
            #     act_probs[4] = 1-sum(act_probs[0:4])
                     
            if availables[idx]==0: idx = -1      
                                         
            # 如果执行的还不错，就用最大概率去执行                             
            # if game.removedlines > self.min_score: 
            #     idx = max_ps_idx     
            # if game.steps < self.limit_count//2:
            if game.next_Pieces_list_len >= len(game.piecehis):
                if self.player>0:
                    if random.random() < 0.75:
                        idx = max_ns_idx
                    else:
                        idx = max_qs_idx
                
            p = 0
            _p = []     
            # if idx == -1 or min(act_qs)<0:    
            if idx == -1 : # or min(act_qs)>=0 :
                # a=1的时候，act 机会均等，>1 强调均值， <1 强调两端
                # 国际象棋 0.3 将棋 0.15 围棋 0.03
                # 取值一般倾向于 a = 10/n 所以俄罗斯方块取 2
                # a = 2       
                # p=0.999**(time.time()-game.start_time)/60     # 每1秒减少0.1的概率
                # if p<0.1: p=0.1
                # p = 0.99**game.removedlines
                # if game.steps < self.limit_count:
                #     p = 0.99
                # else:
                # if game.piececount%2==0:
                #     p = 0.9
                # else:
                #     p = 0.99
                p=0.99
                dirichlet = np.random.dirichlet(2 * np.ones(len(nz_idx)))
                dirichlet_probs = np.zeros_like(act_probs, dtype=np.float64)
                dirichlet_probs[nz_idx] = dirichlet
                
                _act_probs = act_probs * availables
                if np.sum(_act_probs) == 0:
                    _act_probs[nz_idx] = 1.0 / len(nz_idx)
                else:
                    _act_probs = _act_probs / np.sum(_act_probs)
                    
                _p = p*_act_probs + (1.0-p)*dirichlet_probs
                _p = _p / np.sum(_p) 
                idx = np.random.choice(range(ACTONS_LEN), p=_p)
           
            # if (idx == 3 or idx == 0) and random.random()>0.5: # None
            #     idx =  random.choice(nz_idx)
                  
            action = idx
            qval = act_qs[idx]

            if idx!=max_ns_idx:
                print("          random: {} ==> {} v: {} ==> {} p: {} ==> {} probs: {}".format(game.position_to_action_name(max_ns_idx), \
                    game.position_to_action_name(idx), act_qs[max_ns_idx], act_qs[idx], act_probs[max_ns_idx], act_probs[idx], _p))
            #     need_max_ns = "need_max_ns" if self.need_max_ns else ""
            #     need_max_ps = "need_max_ps" if self.need_max_ps else ""
            #     print("\trandom", game.position_to_action_name(max_ps_idx), "==>",  game.position_to_action_name(idx), \
            #           "v:", qval, "p:", p, need_max_ns, need_max_ps, "PS:", act_ps[max_ps_idx], "==>", act_probs[max_ps_idx])  
            if max_ps_idx!=max_ns_idx:
                print("          act_ps: {}, act_probs: {}, old: {}, new: {}".format(act_ps[max_ps_idx], act_probs[max_ps_idx], \
                    game.position_to_action_name(max_ps_idx), game.position_to_action_name(max_ns_idx)))
            acc_ps = 1 if max_ns_idx==max_ps_idx else 0 # np.var(act_probs) #0 if abs(act_ps[idx]-act_probs[idx])>0.4 else 1

            # 将概率转为onehot
            # act_probs = np.zeros_like(act_probs)
            # act_probs[max_qs_idx] = 1
            self.cache[hash(state)] = action


            # 将 act_probs 概率的标准差改为1
            std_act_probs = np.std(act_probs[nz_idx])
            if std_act_probs >1:
                avg_probs = np.mean(act_probs[nz_idx]) 
                act_probs[nz_idx] = (act_probs[nz_idx]-avg_probs) / std_act_probs + avg_probs
                act_probs = np.maximum(act_probs, 0)  # 保证概率不小于0
                act_probs = act_probs / np.sum(act_probs)
            
            # if game.removedlines < 13:
            return action, qval, act_probs, state_v, acc_ps, depth, find_end
        else:
            print("WARNING: game is terminal")

    def __str__(self):
        return "AI {}".format(self.player)