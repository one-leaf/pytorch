# -*- coding: utf-8 -*-
"""
蒙特卡罗树搜索（MCTS）的实现
"""

from math import e
import random
import numpy as np
import copy
import logging
from operator import itemgetter
import heapq
from itertools import count
import time

class TreeNode(object):
    """MCTS树中的节点类。 每个节点跟踪其自身的值Q，先验概率P及其访问次数调整的先前得分u。"""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 子节点 TreeNode
        self._Q = 0  # 节点分数，用于mcts树初始构建时的充分打散（每次叶子节点被最优选中时，节点隔级-leaf_value逻辑，以避免构建树时某分支被反复选中）
        self._n_visits = 0  # 节点被最优选中的次数，用于树构建完毕后的走子选择
        self._P = prior_p  # action概率

    # 扩展新的子节点
    def expand(self, action_priors):
        """把策略函数返回的 [(action,概率)] 列表追加到 child 节点上
            Params：action_priors = 走子策略函数返回的走子概率列表 [(action,概率)]
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    # 从子节点中选择最佳子节点
    def select(self, c_puct):
        """从child中选择最大 action Q+奖励u(P) 的动作
            Params：c_puct = child 搜索深度
            Return: tuple (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    # 计算和返回这个节点的值
    # MCTS计算公式：
    # 传统是    UCT = self._Q/self._n + 2*sqrt(log(root._n)/(1+self._n))
    # 这个是    UCT = self._Q         + 5*self_P*sqrt(_parent._n)/(1+self_n) 
    # 在 leela-zero 中 cfg_puct 中设置的为 0.5 ，这个值如果小就多考虑mcts探索，否则加强概率的影响
    def get_value(self, c_puct):
        """计算并返回当前节点的值
            c_puct:     一个数值，取值范围为（0， inf），调整先验概率和当前路径的权重
            self._P:    action概率
            self._parent._n_visits  父节点的最优选次数
            self._n_visits          当前节点的最优选次数
            self._Q                 当前节点的分数，用于mcts树初始构建时的充分打散
        """
        _u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + _u

    # 反向更新当前节点
    def update(self, leaf_value):
        """更新当前节点的访问次数和叶子节点评估结果
            leaf_value: 从当前玩家的角度看子树的评估值.
        """
        # 访问计数
        self._n_visits += 1
        # 更新 Q, 加上叶子值和当前值的差异的平均数，如果叶子值比本节点价值高，本节点会增高，否则减少.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    # 递归更新当前和其所有的父节点
    def update_recursive(self, leaf_value):
        """同update(), 但是对所有祖先进行递归应用
        """
        # 非root节点时递归update祖先
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    # 检查当前是否已经扩展了
    def is_leaf(self):
        """检查当前是否叶子节点"""
        return self._children == {}

    # 检查当前是否是根节点
    def is_root(self):
        """检查当前是否root节点"""
        return self._parent is None

    def __str__(self):
        if self._parent is None:
            value= 0
        else:
            value = self.get_value(5)
        return "Node - Q: %s, P: %s, N: %s, Value: %s"%(self._Q, self._P, self._n_visits, value) 

class MCTS(object):
    """蒙特卡罗树搜索的实现"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        初始化参数
        policy_value_fn 一个接收board 状态并输出下一步可用动作和概率的函数，
            输出（动作，概率）元组的列表以及[0，1]中的分数，所有分数的合计为1(softmax)
        c_puct 控制搜索速度收敛到最大值的一个权重，取值从 (0, inf) ，这个值越大前面的父节点就会被反向更新的幅度也就越大，
            也就是意味着较高的值会造成更多的依赖之前的步骤
        """
        self._root = TreeNode(None, 1.0)  # 根节点，默认概率值为1
        self._policy = policy_value_fn  # 可走子action及对应概率，这里采用平均概率
        self._c_puct = c_puct  # MCTS child搜索收敛权重
        self._n_playout = n_playout  # 构建MCTS初始树的随机走子步数
        self._keep_best_step = 0     # 不按概率走，直接按最优走

    # 从根节点 root 到子节点执行一次探索过程
    # 1 如果不是叶子，就按子节点的规划执行动作，直到找到叶子
    # 2 获取这一步的所有可能的走法，同时检查游戏有没有结束，如果没有结束，添加到当前节点的父节点
    # 3 继续按随机策略走棋，最终得到当前这一步最后的输赢，如果赢了+1 ，否则-1，没有走完或平局为0
    # 4 按上面得到的价值更新这条线上的所有Q值和选择次数
    def _playout(self, state):
        """
        执行一步随机走子，对应一次MCTS树持续构建过程（选择最优叶子节点->根据走子策略概率扩充mcts树->评估并更新树的最优选次数）
            Params：state盘面 构建过程中会模拟走子，必须传入盘面的copy.deepcopy副本
        """
        # 1.Selection（在树中找到一个最好的值得探索的节点，一般策略是先选择未被探索的子节点，如果都探索过就选择UCB值最大的子节点）
        curr_player = state.current_player
        node = self._root
        # 找到最优叶子节点：递归从child中选择并执行最大 动作Q+奖励u(P) 的动作
        while (1):
            if node.is_leaf():
                break

            # 从child中选择最优action
            action, node = node.select(self._c_puct)
            # 执行action走子
            state.step(action)
            
        # 2.Expansion（就是在前面选中的子节点中走一步创建一个新的子节点。一般策略是随机自行一个操作并且这个操作不能与前面的子节点重复）
        # 走子策略返回的[(action,概率)]list
        action_probs, leaf_value = self._policy(state)
        # 检查游戏是否有赢家
        end, score = state.game_end()
        if not end:  # 没有结束时，把走子策略返回的[(action,概率)]list加载到mcts树child中
            node.expand(action_probs)

        # 3.Simulation（在前面新Expansion出来的节点开始模拟游戏，直到到达游戏结束状态，这样可以收到到这个expansion出来的节点的得分是多少）
        # 使用快速随机走子评估此叶子节点继续往后走的胜负（state执行快速走子）
        leaf_value = self._evaluate_rollout(state)
        # 4.Backpropagation（把前面expansion出来的节点得分反馈到前面所有父节点中，更新这些节点的quality value和visit times，方便后面计算UCB值）
        # 递归更新当前节点及所有父节点的最优选中次数和Q分数（最优选中次数是累加的）
        node.update_recursive(leaf_value)

    # 从根节点 root 到子节点执行一次探索过程
    # 这个不同于上面，上面的是纯mcts,后面多一步对当前动作进行评估的过程，这个是直接用网络来估测当前可以步数的价值
    def _playout_network(self, state):
        """
        执行一步走子，对应一次MCTS树持续构建过程（选择最优叶子节点->根据模型走子策略概率扩充mcts树->评估并更新树的最优选次数）
            Params：state盘面 构建过程中会模拟走子，必须传入盘面的copy.deepcopy副本
        """
        node = self._root
        reward = 0
        # 找到最优叶子节点：递归从child中选择并执行最大 动作Q+奖励u(P) 的动作
        while (1):
            if node.is_leaf():
                break

            # 从child中选择最优action
            action = None
            #for act in node._children:
            #    if node._children[act]._n_visits == 0:
            #        action, node = act, node._children[act]
            #        break

            if action is None:
                action, node = node.select(self._c_puct)
            # 执行action走子
            _, reward = state.step(action)

        # 检查游戏是否有赢家
        end, score = state.game_end()
        # if reward==0:# and state.state!=0:
        #     reward, _ ,_ = state.checkActionisBest(include_fallpiece=True)
        if not end:  # 没有结束时，把走子策略返回的[(action,概率)]list加载到mcts树child中 ，同时降低了 leaf_value 的权重
            # 使用训练好的模型策略评估此叶子节点，返回[(action,概率)]list 以及当前玩家的后续走子胜负
            action_probs, leaf_value = self._policy(state)
            # 如果没有结束，顺便加上中途检测得分
            # leaf_value = reward
            node.expand(action_probs)
        else:
            leaf_value = -1   # 如果游戏结束，得分就是-1，尽量不结束游戏

        # 早期完全使用修正,到局部修正到最后的结束时再判定
        # 这里的评价不一定是对的，所以只能给一个很小的值参考
        # if state.state!=0:
        #     v = state.checkActionisBest(include_fallpiece=False)            
        #     # leaf_value = v
        #     # leaf_value = -1*np.log(-1*v) 
        #     leaf_value += v*-1e-5

            # print("leaf_value", leaf_value)

        # 给熵加一点点的支持
        if state.state==2:
            leaf_value = reward
            print(reward)

        #     leaf_value -= np.log(state.getTransCount())
        # 递归更新当前节点及所有父节点的最优选中次数和Q分数,因为得到的是本次的价值
        node.update_recursive(leaf_value)

        # if reward>0:
        #     # print("Oh Ye!!! get a reward!!! reward:", reward)
        #     _node=node
        #     while _node._parent:
        #         for ac in _node._parent._children:
        #             if _node._parent._children[ac]==_node:
        #                 self._keep_best_step += 1 
        #                 # print(self._keep_best_step, ":", "action:", ac, _node)
        #                 break
        #         _node = _node._parent


    def update_root_with_action(self, action):
        """根据action更新根节点"""
        if action!=None: #action in self._root._children:
            self._root = self._root._children[action]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
            self._keep_best_step = 0

    def _evaluate_rollout(self, state, limit=1000):
        """使用随机快速走子策略评估叶子节点
            Params：
                state 当前盘面
                limit 随机走子次数
            Return：如果当前玩家获胜返回+1
                    如果对手获胜返回-1
                    如果平局返回0
        """
        curr_player = state.current_player
        score = -1
        for i in range(limit):  # 随机快速走limit次，用于快速评估当前叶子节点的优略
            end, score = state.game_end()
            if end:
                break
            # 给棋盘所有可落子位置随机分配概率，并取其中最大概率的action移动
            action_probs = MCTS.rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.step(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if score == -1:  # tie平局
            return 0
        else:
            return (1.0 if score >0  else -1.0)

    @staticmethod
    def rollout_policy_fn(state):
        """给棋盘所有可落子位置随机分配概率"""
        availables = state.availables
        action_probs = np.random.rand(len(availables))
        return zip(availables, action_probs)

    @staticmethod
    def policy_value_fn(state):
        """给棋盘所有可落子位置分配默认平均概率 [(0, 0.015625), (action, probability), ...], 0"""
        availables = state.availables
        action_probs = np.ones(len(availables)) / len(availables)
        return zip(availables, action_probs), 0

    @staticmethod
    def softmax(x):
        """softmax"""
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

    # 按概率返回当前状态下的动作及其概率，构建所有的树，默认10000局
    # 这个不同于 get_action ，这个是 mtcs + AI 的特殊走法 
    def get_action_probs(self, state, temp=1e-3):
        """
        构建模型网络MCTS初始树，并返回所有action及对应模型概率
            Params：
                state: 当前游戏盘面
                temp：温度参数  控制探测水平，范围(0,1]
            Return: 所有action及对应概率
        """
        # for n in count():
        for n in range(self._n_playout):
            # print("\r_n_playout： {:.2f}%".format(n*100 / self._n_playout), end='')
            state_copy = copy.deepcopy(state)
            self._playout_network(state_copy)

            # 如果只有一个动作
            if len(self._root._children)==1:
                break
            # 为了提高学习效率如果有探索的标准差大于100，直接放弃探索,返回。
            # if n>0:
            #     act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
            #     acts, visits = zip(*act_visits)
            #     # 如果只有一个选项，直接返回
            #     if len(acts)==1: break
            #     var = np.var(visits)
            #     if var>100:
            #         break
            
                # if n>=self._n_playout:
                #     # 如果得分为负数，多算2倍，争取找出一个优解
                #     # value = self._root._children[acts[idx]].get_value(5)
                #     # if value>0 or n>self._n_playout*2:
                #     break

        # 分解出child中的action和最优选访问次数
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        # 打印中间信息
        if state.piecesteps==0:
            depth = self.max_depth_tree()
            info={"shape":state.fallpiece["shape"], "depth":depth}
            # if depth>100:
            #     self.print_tree()
            # info={"shape":state.fallpiece["shape"]}
            for idx in sorted(range(len(visits)), key=visits.__getitem__)[::-1]:
                value = self._root._children[acts[idx]].get_value(0.5)
                info[acts[idx]] = (visits[idx], round(value, 2))
            # state.print(add_fallpiece=False)
            # print(state.checkActionisBest(include_fallpiece=True))
            print("steps:",state.steps,"_n_playout:", n, "info:", info)
            # self.print_tree()
        # softmax概率，先用log(visites)，拉平差异，再乘以一个权重，这样给了一个可以调节的参数，
        # temp 越小，导致softmax的越肯定，也就是当temp=1e-3时，基本上返回只有一个1,其余概率都是0; 训练的时候 temp=1
        # act_probs = MCTS.softmax((1/temp) * np.log(np.array(visits) + 1e-10))
        m = np.power(np.array(visits), 1./temp)
        act_probs = m/np.sum(m)        
        return acts, act_probs

    def max_depth_tree(self, node=None):
        if node==None:
            node=self._root
        l=[0]
        for act in node._children:
            l.append(self.max_depth_tree(node._children[act])+1)
        return max(l)

    def print_tree(self, node=None, depth=0):
        if depth == 0:
            print("root")
            node=self._root
        for act in node._children:
            print(str(depth) + " +-- " + str(act))
            new_node = node._children[act]
            self.print_tree(new_node, depth +1)


    # 按访问次数返回当前状态下的动作及其概率，构建所有的树，默认10000局
    # 这个是 mcts 的标准方法
    def get_action(self, state):
        """
        构建纯MCTS初始树(节点分布充分)，并返回child中访问量最大的action
            state: 当前游戏盘面
            Return: 构建的树中访问量最大的action
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def __str__(self):
        return "MCTS"


class MCTSPurePlayer(object):
    """基于纯MCTS的player"""

    def __init__(self, c_puct=5, n_playout=2000):
        """初始化参数"""
        self.mcts = MCTS(MCTS.policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        """指定MCTS的playerid"""
        self.player = p

    def reset_player(self):
        """更新根节点:根据最后action向前探索树"""
        self.mcts.update_root_with_action(None)

    def get_action(self, state):
        """计算下一步走子action"""
        # 构建纯MCTS初始树(节点分布充分)，并返回child中访问量最大的action
        action = self.mcts.get_action(state)
        # 更新根节点:根据最后action向前探索树
        self.mcts.update_root_with_action(None)
        print("MCTS:", action)
        return action

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

    def reset_player(self):
        """根据最后action向前探索树（通过_root保存当前探索位置）"""
        self.mcts.update_root_with_action(None)

    def get_action(self, state, temp=1e-3, return_prob=0):
        """计算下一步走子action"""
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(state.actions_num)
        acts, act_probs = self.mcts.get_action_probs(state, temp)
        if len(acts) > 0:  # 可用动作>0
            # 使用默认的temp = 1e-3，它几乎相当于选择具有最高概率的移动 ，训练的时候 temp = 1
            move_probs[list(acts)] = act_probs
            if self._is_selfplay:  # 自我对抗
                # 添加Dirichlet Noise进行探索（自我训练所需）
                # dirichlet噪声参数中的p 0.3：一般按照反比于每一步的可行move数量设置，所以棋盘扩大或改围棋之后这个参数需要减小（此值设置过大容易出现在自我对弈的训练中陷入到两方都只进攻不防守的困境中无法提高）
                # dirichlet噪声是分布的分布，sum为1，参数越大，分布越均匀，参数越小越集中
                # 给定的是一个均匀分布，则参数越小，方差越大，扰动就越大
                # if max(act_probs)>0.99:
                #     p = 1.
                # else:
                #     p = 0.9  
                # p = state.steps/200
                # if p>0.9: p=0.9
                idx = np.argmax(act_probs)                    
                if act_probs[idx] > 0.8:# self.mcts._keep_best_step>0:
                    action = acts[idx]
                    # print(" - ", self.mcts._keep_best_step, ":", "action:", action)
                    # self.mcts._keep_best_step -= 1
                else:
                    # 如果是前15个方块的前5步，有一半的几率乱走,或5%的可能乱走                   
                    # p=0.75
                    # dirichlet = np.random.dirichlet(0.3*np.ones(len(act_probs)))
                    # action = np.random.choice(acts, p= p*act_probs + (1-p)*dirichlet) 
                    action = np.random.choice(acts, p= act_probs) 
                    if action!=acts[idx]:
                        print(" pices_step:", state.piecesteps, acts[idx], act_probs[idx], "==>", action, act_probs[acts.index(action)])
                # 更新根节点并重用搜索树
                self.mcts.update_root_with_action(action)
                # self.mcts.update_root_with_action(None)
            else:  # 正式玩
                action = np.random.choice(acts, p=act_probs)
                # 更新根节点:根据最后action向前探索树
                self.mcts.update_root_with_action(None)
                # 打印AI走子信息
                print("AI:", action, act_probs[acts.index(action)]) 
            # print("AI:", action)
            if return_prob:
                return action, move_probs
            else:
                return action
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTSPlayer {}".format(self.player)