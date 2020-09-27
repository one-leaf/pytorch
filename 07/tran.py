import os, glob, pickle

from time import time
from model import PolicyValueNet
import logging
from agent import Agent
from mcts import MCTSPlayer

import sys, time

from itertools import count
from collections import deque
from collections import namedtuple
import os, math, random

import numpy as np
import torch

# 定义游戏的动作
GAME_ACTIONS_NUM = 4 
GAME_WIDTH, GAME_HEIGHT = 10, 20

# 定义游戏的保存文件名和路径
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, './data/')
if not os.path.exists(data_dir): os.makedirs(data_dir)
model_dir = os.path.join(curr_dir, './model/')
if not os.path.exists(model_dir): os.makedirs(model_dir)
model_file =  os.path.join(model_dir, 'model.pth')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, buffer_size):
        # 训练数据存放路径
        self.data_dir = data_dir                
        # 训练数据最大保存个数
        self.buffer_size = buffer_size

        # 当前训练数据索引保存文件
        self.data_index_file = os.path.join(data_dir, 'index.txt')
        # 当前训练数据索引
        self.curr_game_batch_num = 0        
        self.load_game_batch_num()

        # 当前数据训练文件
        self.file_list = []
        self.load_game_files()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        # 状态，步骤的概率，最终得分
        state, mcts_prob, winner = pickle.load(open(filename, "rb"))
        state = torch.from_numpy(state).float()
        mcts_prob = torch.from_numpy(mcts_prob).float()
        winner = torch.as_tensor(winner).float()
        return state, mcts_prob, winner

    def load_game_files(self):
        files = glob.glob(os.path.join(self.data_dir, "*.pkl"))
        files = sorted(files, key=lambda x: os.path.getmtime(x))
        for filename in files:
            self.file_list.append(filename)

    def save_game_batch_num(self):
        with open(self.data_index_file, "w") as f:
            f.write(str(self.curr_game_batch_num))

    def load_game_batch_num(self):
        if os.path.exists(self.data_index_file):
            self.curr_game_batch_num = int(open(self.data_index_file, 'r').read().strip())

    # 保存新的训练样本，但不参与到本次训练，等下一次训练加载
    def save(self, obj):
        with self._save_lock:
            # 文件名为buffer取余，循环保存
            filename = "%s.pkl" % self.curr_game_batch_num % self.buffer_size
            savefile = os.path.join(self.data_dir, filename)
            pickle.dump(obj, open(savefile, "wb"))
            self.curr_game_batch_num += 1
            self.save_game_batch_num()

    def curr_size(self):
        return len(self.file_list)

class Train():
    def __init__(self):
        self.game_batch_num = 1000000  # selfplay对战次数
        self.batch_size = 512     # data_buffer中对战次数超过n次后开始启动模型训练

        # training params
        self.learn_rate = 1e-4
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1.0  # the temperature param
        self.n_playout = 100  # 每个动作的模拟次数
        self.buffer_size = 100000  # cache对战记录个数
        self.play_batch_size = 2 # 每次自学习次数
        self.epochs = 5  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        self.c_puct = 5  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度
        self.policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)


    def collect_selfplay_data(self):
        """收集自我对抗数据用于训练"""       
        # 使用MCTS蒙特卡罗树搜索进行自我对抗
        logging.info("TRAIN Self Play starting ...")
        # 游戏代理
        agent = Agent()

        # 创建使用策略价值网络来指导树搜索和评估叶节点的MCTS玩家
        mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)
        # 开始下棋
        winer, play_data = agent.start_self_play(mcts_player, temp=self.temp)
        play_data = list(play_data)[:]
        episode_len = len(play_data)

        # 把翻转棋盘数据加到数据集里
        logging.info("TRAIN Self Play end. length:%s saving ..." % episode_len)
        # 保存对抗数据到data_buffer
        for obj in play_data:
            self.dataset.save(obj)

        agent.print()                   

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
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)
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
        logging.info(("TRAIN kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{:.5f},var_old:{:.5f},var_new:{:.5f}"
                      ).format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy  

    def run(self):
        """启动训练"""
        try:
            print("start data loader")
            self.dataset = Dataset(data_dir, self.buffer_size)
            print("end data loader")

            step = 0
            while self.dataset.curr_size() < self.batch_size*self.epochs:
                logging.info("TRAIN Batch:{} starting".format(step + 1,))
                self.collect_selfplay_data()
                logging.info("TRAIN Batch:{} end".format(step + 1,))
                step += 1

            training_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,)

            for i, data in enumerate(training_loader):  # 计划训练批次
                # 使用对抗数据重新训练策略价值网络模型
                loss, entropy = self.policy_update(data, self.epochs)
                # 每n个batch检查一下当前模型胜率

                if (i+1) % (int(self.dataset.curr_size() ** 0.3)) == 0:
                    self.policy_value_net.save_model(model_file)
                    # 收集自我对抗数据
                    for _ in range(self.play_batch_size):
                        self.collect_selfplay_data()
                    logging.info("TRAIN {} self-play end, size: {}".format(i, self.dataset.curr_size()))
                    
    
        except KeyboardInterrupt:
            logging.info('quit')

if __name__ == '__main__':
    # train
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    training = Train()
    training.run()

