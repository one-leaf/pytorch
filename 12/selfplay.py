import os, glob, pickle

from time import time
from model import PolicyValueNet
from agent import Agent, ACTIONS
from mcts import MCTSPlayer

import sys, time, json, datetime

from itertools import count
from collections import deque
from collections import namedtuple
import os, math, random, uuid

import numpy as np
import copy

# 定义游戏的动作
GAME_ACTIONS_NUM = len(ACTIONS) 
GAME_WIDTH, GAME_HEIGHT = 10, 20

# 定义游戏的保存文件名和路径
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, './data/')
if not os.path.exists(data_dir): os.makedirs(data_dir)

data_wait_dir = os.path.join(curr_dir, './data/wait/')
if not os.path.exists(data_wait_dir): os.makedirs(data_wait_dir)

model_dir = os.path.join(curr_dir, './model/')
if not os.path.exists(model_dir): os.makedirs(model_dir)
model_file =  os.path.join(model_dir, 'model-mlp.pth')

class Train():
    def __init__(self):
        self.game_batch_num = 1000000  # selfplay对战次数
        self.batch_size = 512     # data_buffer中对战次数超过n次后开始启动模型训练

        # training params
        self.learn_rate = 1e-4
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # MCTS的概率参数，越大越不肯定，训练时1，预测时1e-3
        self.n_playout = 256  # 每个动作的模拟战记录个数
        self.play_batch_size = 5 # 每次自学习次数
        self.buffer_size = 1000000  # cache对次数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        self.c_puct = 0.1  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5
        self.policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)

    def collect_selfplay_data(self):
        """收集自我对抗数据用于训练"""       
        print("TRAIN Self Play starting ...")

        # 获取历史训练数据
        jsonfile = os.path.join(data_dir, "result.json")
        if os.path.exists(jsonfile):
            result=json.load(open(jsonfile,"r"))
        else:
            result={}
            result={"agent":0, "reward":[], "pieces":[], "qvals":[], "QVal":0}
        if "curr" not in result:
            result["curr"]={"reward":0,"pieces":0,"agent":0}
        if "best" not in result:
            result["best"]={"reward":0,"pieces":0,"agent":0}

        hisQval=result["QVal"]
        print("QVal:",hisQval)

        # 游戏代理
        agent = Agent()

        max_game_num = 2
        agentcount, agentreward, piececount, agentscore = 0, 0, 0, 0
        game_states, game_vals, game_mcts_probs = [], [], [] 

        borads = []
        game_num = 0
        can_exit_flag = False
        for game_idx in count():
            game_num += 1
            player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

            _states, _probs, _masks, _rewards, _qvals = [],[],[],[],[]
            game = copy.deepcopy(agent)

            if game_idx==0:
                game.show_mcts_process=True

            for i in count():               
                _states.append(game.current_state())
                                
                if game_idx == game_num-1:
                    action, move_probs = player.get_action(game, temp=self.temp, return_prob=1, need_random=True) 
                else: 
                    action, move_probs = player.get_action(game, temp=self.temp, return_prob=1, need_random=True) 
               
                _, reward = game.step(action)

                # 这里的奖励是消除的行数
                if reward > 0:
                    _reward = reward * 10
                    print("#"*50, reward, "#"*50)
                else:
                    _reward = 0

                # 方块的个数越多越好
                if game.terminal:
                    _reward = game.getNoEmptyCount() + game.score * 10     
                    if _reward > hisQval: can_exit_flag = True         

                _probs.append(move_probs)
                _rewards.append(_reward)
                _masks.append(1-game.terminal)

                if game.terminal:
                    for step in reversed(range(len(_states))):
                        Qval = _rewards[step]
                        _qvals.insert(0, Qval)

                    print(game_idx, 'reward:', game.score, "Qval:", _rewards[-1], 'len:', len(_qvals), "piececount:", game.piececount)
                    agentcount += 1
                    agentscore += game.score
                    agentreward += _reward
                    piececount += game.piececount

                    if _reward>result["best"]["reward"]:
                        result["best"]["reward"] = _reward
                        result["best"]["pieces"] = game.piececount
                        result["best"]["score"] = game.score
                        result["best"]["agent"] = result["agent"]+agentcount
                        
                    break

            game_states.append(_states)
            game_vals.append(_qvals)
            game_mcts_probs.append(_probs)

            game.print()
            borads.append(game.board)

            # 如果训练次数超过了最大次数，并且最大得分值超过了平均得分值，则停止训练
            if game_num >= max_game_num and can_exit_flag: break

            # 如果训练次数超过了最大次数的3倍，则直接终止训练
            if game_num >= max_game_num*3: break


        # 打印borad：
        from game import blank 
        for y in range(agent.height):
            line=""
            for b in borads:
                line+="| "
                for x in range(agent.width):
                    if b[x][y]==blank:
                        line+="  "
                    else:
                        line+="%s " % b[x][y]
            print(line)
        print((" "+" -"*agent.width+" ")*len(borads))

        avg_value = []
        for game_idx in range(game_num):
            for i in reversed(range(len(game_vals[game_idx])-1)):
                game_vals[game_idx][i] += game_vals[game_idx][i+1]*1 # 0.999  
            avg_value.extend(game_vals[game_idx])
            print(len(game_vals[game_idx]), ":", *game_vals[game_idx][:3], "...", *game_vals[game_idx][-3:])

        curr_avg_value = sum(avg_value)/len(avg_value)
        curr_std_value = np.std(avg_value)
        print("avg_value:", curr_avg_value, "std_value:", curr_std_value)

        if hisQval==0:
            avg_value = curr_avg_value            
        else:
            avg_value = hisQval*0.999 + curr_avg_value*0.001

        result["QVal"] = avg_value
               
        states, values, mcts_probs= [], [], []
        for j in range(game_num):
            for o in game_states[j]: states.append(o)
            for o in game_mcts_probs[j]: mcts_probs.append(o)
            normalize_vals = []
            for o in game_vals[j]: 
                # 这里考虑还是用所有局的平均值作为衡量标准，而不是全部的平均值
                # 标准化的标准差为0.5
                v = (o-curr_avg_value)/(curr_std_value*2)
                if v>1: v=1
                if v<-1: v=-1
                normalize_vals.append(v)            
            values.extend(normalize_vals)
            print(*normalize_vals[:5], "...", *normalize_vals[-5:])

        assert len(states)==len(values)
        assert len(states)==len(mcts_probs)

        print("TRAIN Self Play end. length:%s value sum:%s saving ..." % (len(states),sum(values)))

        # 保存对抗数据到data_buffer
        for obj in zip(states, mcts_probs, values):
            filename = "{}.pkl".format(uuid.uuid1())
            savefile = os.path.join(data_wait_dir, filename)
            pickle.dump(obj, open(savefile, "wb"))

        result["agent"] += agentcount
        result["curr"]["reward"] += agentscore
        result["curr"]["pieces"] += piececount
        result["curr"]["agent1000"] += agentcount
        result["curr"]["agent100"] += agentcount

        agent = result["agent"]
        if result["curr"]["agent100"]>100:
            result["reward"].append(round(result["curr"]["reward"]/result["curr"]["agent"],2))
            result["pieces"].append(round(result["curr"]["pieces"]/result["curr"]["agent"],2))
            result["qvals"].append(round(avg_value,2))
            result["curr"]["agent100"] -= 100 
            if len(result["reward"])>250:
                result["reward"].remove(result["reward"][0])
            if len(result["pieces"])>250:
                result["pieces"].remove(result["pieces"][0])
            if len(result["qvals"])>250:
                result["qvals"].remove(result["qvals"][0])

        if result["curr"]["agent1000"]>1000:
            result["curr"]={"reward":0,"pieces":0,"agent1000":0,"agent100":0}

            newmodelfile = model_file+"_"+str(agent)
            if not os.path.exists(newmodelfile):
                self.policy_value_net.save_model(newmodelfile)

        json.dump(result, open(jsonfile,"w"), ensure_ascii=False)

    def policy_update(self, sample_data, epochs=1):
        """更新策略价值网络policy-value"""
        # 训练策略价值网络
        # 随机抽取data_buffer中的对抗数据
        # mini_batch = self.dataset.loadData(sample_data)
        state_batch, mcts_probs_batch, winner_batch = sample_data
        # # for x in mini_batch:
        # #     print("-----------------")
        # #     print(x)
        # # state_batch = [data[0] for data in mini_batch]
        # # mcts_probs_batch = [data[1] for data in mini_batch]
        # # winner_batch = [data[2] for data in mini_batch]

        # print(state_batch)

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)  
        for i in range(epochs):
            loss, v_loss, p_loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            # 散度计算：
            # D(P||Q) = sum( pi * log( pi / qi) ) = sum( pi * (log(pi) - log(qi)) )
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # 如果D_KL跑偏则尽早停止
                break

        # 自动调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # 如果学习到了，explained_var 应该趋近于 1，如果没有学习到也就是胜率都为很小值时，则为 0
        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        # entropy 信息熵，越小越好
        print(("TRAIN kl:{:.5f},lr_multiplier:{:.3f},v_loss:{:.5f},p_loss:{:.5f},entropy:{:.5f},var_old:{:.5f},var_new:{:.5f}"
                      ).format(kl, self.lr_multiplier, v_loss, p_loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy  

    def run(self):
        """启动训练"""
        try:
            self.collect_selfplay_data()    
        except KeyboardInterrupt:
            print('quit')

if __name__ == '__main__':
    print('start training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    training = Train()
    training.run()
    print('end training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

