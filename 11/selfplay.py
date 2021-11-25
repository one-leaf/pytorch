import os, glob, pickle

from time import time
from model import PolicyValueNet
from agent import Agent, ACTIONS

import sys, time, json

from itertools import count
from collections import deque
from collections import namedtuple
import os, math, random, uuid

import numpy as np
import torch

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
        self.n_playout = 500  # 每个动作的模拟战记录个数
        self.play_batch_size = 5 # 每次自学习次数
        self.buffer_size = 500000  # cache对次数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        self.c_puct = 5  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5
        self.policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)

    def collect_selfplay_data(self):
        """收集自我对抗数据用于训练"""       
        print("TRAIN Self Play starting ...")
        # 游戏代理
        agent = Agent()

        # 开始下棋
        agentcount, reward, piececount, keys, play_data = agent.start_self_play(self.policy_value_net)
        
        play_data = list(play_data)[:]
        episode_len = len(play_data)

        print("TRAIN Self Play end. length:%s saving ..." % episode_len)
        # 保存对抗数据到data_buffer
        for i, obj in enumerate(play_data):
            filename = "{}.pkl".format(uuid.uuid1())
            savefile = os.path.join(data_wait_dir, filename)
            pickle.dump(obj, open(savefile, "wb"))

        jsonfile = os.path.join(data_dir, "result.json")
        if os.path.exists(jsonfile):
            result=json.load(open(jsonfile,"r"))
        else:
            result={}
            result={"agent":0,"reward":[],"pieces":[]}
            result["curr"]={"reward":0,"pieces":0,"agent":0}

        result["agent"] += agentcount
        result["curr"]["reward"] += reward
        result["curr"]["pieces"] += piececount
        result["curr"]["agent"] += agentcount

        agent = result["agent"]
        if agent%100==0:
            result["reward"].append(round(result["curr"]["reward"]/result["curr"]["agent"],2))
            result["pieces"].append(round(result["curr"]["pieces"]/result["curr"]["agent"],2))
            if len(result["reward"])>100:
                result["reward"].remove(min(result["reward"]))
            if len(result["pieces"])>100:
                result["pieces"].remove(min(result["pieces"]))
        if result["curr"]["agent"]>1000:
            result["curr"]={"reward":0,"pieces":0,"agent":0}

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
    # train
    training = Train()
    training.run()

