import os, glob, pickle

from time import time
from model import PolicyValueNet
import logging
from agent import Agent, ACTIONS
from mcts import MCTSPlayer

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
        # 文件名为buffer取余，循环保存
        filename = "{}.pkl".format(uuid.uuid1())
        # filename = "{}.pkl".format(self.curr_game_batch_num % self.buffer_size,)
        savefile = os.path.join(data_wait_dir, filename)
        pickle.dump(obj, open(savefile, "wb"))
        # self.curr_game_batch_num += 1
        # self.save_game_batch_num()
        
    def curr_size(self):
        return len(self.file_list)

class Train():
    def __init__(self):
        self.game_batch_num = 1000000  # selfplay对战次数
        self.batch_size = 512     # data_buffer中对战次数超过n次后开始启动模型训练

        # training params
        self.learn_rate = 1e-5
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # MCTS的概率参数，越大越不肯定，训练时1，预测时1e-3
        self.n_playout = 64  # 每个动作的模拟战记录个数
        self.play_batch_size = 5 # 每次自学习次数
        self.buffer_size = 300000  # cache对次数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        self.c_puct = 0.1  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5
        self.policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)

    def get_equi_data(self, play_data):
        """
        通过翻转增加数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            extend_data.append((state, mcts_porb, winner))
            # 水平翻转
            equi_state = np.array([np.fliplr(s) for s in state])
            equi_mcts_prob = mcts_porb[[0,2,1,3]]
            extend_data.append((equi_state, equi_mcts_prob, winner))
        return extend_data

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
        # play_data = self.get_equi_data(play_data)
        logging.info("TRAIN Self Play end. length:%s saving ..." % episode_len)
        # 保存对抗数据到data_buffer
        for obj in play_data:
            filename = "{}.pkl".format(uuid.uuid1())
            savefile = os.path.join(data_wait_dir, filename)
            pickle.dump(obj, open(savefile, "wb"))
            # self.dataset.save(obj)
        
        jsonfile = os.path.join(data_dir, "result.json")
        if os.path.exists(jsonfile):
            result=json.load(open(jsonfile,"r"))
        else:
            result={"reward":0,"steps":0,"agent":0}
        if agent.score>0:
            result["reward"] = result["reward"] + 1
        result["steps"] = result["steps"] + agent.piececount
        result["agent"] = result["agent"] + 1
        if result["agent"]>0 and result["agent"]%1000==0:
            for key in list(result.keys()):
                if key.isdigit():
                    c = int(key)
                    if c%1000!=0:
                        del result[key]

        if result["agent"]>0 and result["agent"]%100==0:
            result[str(result["agent"])]={"reward":result["reward"]/result["agent"],
                                            "steps":result["steps"]/result["agent"]}
        
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
        logging.info(("TRAIN kl:{:.5f},lr_multiplier:{:.3f},v_loss:{:.5f},p_loss:{:.5f},entropy:{:.5f},var_old:{:.5f},var_new:{:.5f}"
                      ).format(kl, self.lr_multiplier, v_loss, p_loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy  

    def run(self):
        """启动训练"""
        try:
            # print("start data loader")
            # self.dataset = Dataset(data_dir, self.buffer_size)
            # print("end data loader")

            # step = 0
            # # 如果训练数据一半都不到，就先攒训练数据
            # if self.dataset.curr_game_batch_num/self.dataset.buffer_size<0.5:
            #     for _ in range(8):
            #         logging.info("TRAIN Batch:{} starting".format(self.dataset.curr_game_batch_num,))
            #         # n_playout=self.n_playout
            #         # self.n_playout=8
            #         self.collect_selfplay_data()
            #         # self.n_playout=n_playout
            #         logging.info("TRAIN Batch:{} end".format(self.dataset.curr_game_batch_num,))
            #         step += 1

            # training_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,)

            # for i, data in enumerate(training_loader):  # 计划训练批次
            #     # 使用对抗数据重新训练策略价值网络模型
            #     loss, entropy = self.policy_update(data, self.epochs)

            # self.policy_value_net.save_model(model_file)
            # 收集自我对抗数据
            # for _ in range(self.play_batch_size):
            self.collect_selfplay_data()
            # logging.info("TRAIN {} self-play end, size: {}".format(self.dataset.curr_game_batch_num, self.dataset.curr_size()))
                    
    
        except KeyboardInterrupt:
            logging.info('quit')

if __name__ == '__main__':
    # train
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    training = Train()
    training.run()

