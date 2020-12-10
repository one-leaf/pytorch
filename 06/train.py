from shutil import copyfile
from torch import batch_norm, sqrt
from model import PolicyValueNet  
from mcts import MCTSPurePlayer, MCTSPlayer
from agent import Agent
import os, glob, pickle
import sys, time
import random
import logging
import numpy as np
from collections import defaultdict, deque
import torch
from threading import Thread, Lock

curr_dir = os.path.dirname(os.path.abspath(__file__))
size = 15  # 棋盘大小
n_in_row = 5  # 几子连线

data_dir = os.path.join(curr_dir, './data/%s_%s/'%(size,n_in_row))
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

data_wait_dir = os.path.join(curr_dir, './data/%s_%s_wait/'%(size,n_in_row))
if not os.path.exists(data_wait_dir):
    os.makedirs(data_wait_dir)

model_dir = os.path.join(curr_dir, './model/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_file =  os.path.join(model_dir, 'model_%s_%s.pth'%(size,n_in_row))
best_model_file =  os.path.join(model_dir, 'best_model_%s_%s.pth'%(size,n_in_row))

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    # trans_count 希望训练的总记录数，为 训练轮次 * Batch_Size
    # max_keep_size 最多保存的训练样本数
    def __init__(self, data_dir, max_keep_size):
        self.data_dir = data_dir
        self.max_keep_size = max_keep_size
        self.index = 0
        self.data_index_file = os.path.join(data_dir, 'index.txt')
        self.file_list = deque(maxlen=max_keep_size)        
        self._save_lock = Lock()
        self.load_index()
        self.copy_wait_file()
        self.load_game_files()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
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

    def save_index(self):
        with open(self.data_index_file, "w") as f:
            f.write(str(self.index))

    def load_index(self):
        if os.path.exists(self.data_index_file):
            self.index = int(open(self.data_index_file, 'r').read().strip())

    def copy_wait_file(self):
        movefiles=os.listdir(data_wait_dir)
        # 等待一秒钟，防止有数据还在写入
        time.sleep(1)
        for i, fn in enumerate(movefiles):
            filename = "{}.pkl".format(self.index % self.max_keep_size,)
            savefile = os.path.join(self.data_dir, filename)
            if os.path.exists(savefile): os.remove(savefile)
            os.rename(os.path.join(data_wait_dir,fn), savefile)
            self.index += 1
            self.save_index()        
        print("mv %s files to train"%len(movefiles))

    def save(self, obj):
        with self._save_lock:
            filename = "{}.pkl".format(self.index % self.max_keep_size,)
            savefile = os.path.join(self.data_dir, filename)
            pickle.dump(obj, open(savefile, "wb"))
            self.index += 1
            self.save_index()

class FiveChessTrain():
    def __init__(self):
        self.policy_evaluate_size = 20  # 策略评估胜率时的模拟对局次数
        self.batch_size = 512  # 训练一批数据的长度
        self.max_keep_size = 500000  # 保留最近对战样本个数 平均一局大约400~600个样本, 也就是包含了最近1000次对局数据

        # 训练参数
        self.learn_rate = 1e-5
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # 概率缩放程度，实际预测0.01，训练采用1
        self.n_playout = 600  # 每个动作的模拟次数
        self.play_batch_size = 1 # 每次自学习次数
        self.epochs = 1  # 重复训练次数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        
        # 纯MCTS的模拟数，用于评估策略模型
        self.pure_mcts_playout_num = 4000 # 用户纯MCTS构建初始树时的随机走子步数
        self.c_puct = 1  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5

        if os.path.exists(model_file):
            # 使用一个训练好的策略价值网络
            self.policy_value_net = PolicyValueNet(size, model_file=model_file)
        else:
            # 使用一个新的的策略价值网络
            self.policy_value_net = PolicyValueNet(size)

        print("start data loader")
        self.dataset = Dataset(data_dir, self.max_keep_size)
        print("dataset len:",len(self.dataset),"index:",self.dataset.index)
        print("end data loader")

    def policy_update(self, sample_data, epochs=1):
        """更新策略价值网络policy-value"""
        # 训练策略价值网络
        state_batch, mcts_probs_batch, winner_batch = sample_data

        # old_probs, old_v = self.policy_value_net.policy_value(state_batch)  
        for i in range(epochs):
            loss, v_loss, p_loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)
            # new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            # 散度计算：
            # D(P||Q) = sum( pi * log( pi / qi) ) = sum( pi * (log(pi) - log(qi)) )
            # kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            # if kl > self.kl_targ * epochs:  # 如果D_KL跑偏则尽早停止
            #     break

        # 自动调整学习率
        # if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
        #     self.lr_multiplier /= 1.5
        # elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
        #     self.lr_multiplier *= 1.5
        # 如果学习到了，explained_var 应该趋近于 1，如果没有学习到也就是胜率都为很小值时，则为 0
        # explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        # explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        # entropy 信息熵，越小越好
        # logging.info(("TRAIN kl:{:.5f},lr_multiplier:{:.3f},v_loss:{:.5f},p_loss:{:.5f},entropy:{:.5f},var_old:{:.5f},var_new:{:.5f}"
        #               ).format(kl, self.lr_multiplier, v_loss, p_loss, entropy, explained_var_old, explained_var_new))
        logging.info(("TRAIN v_loss:{:.5f},p_loss:{:.5f},entropy:{:.5f}").format(v_loss, p_loss, entropy))
        return loss, entropy

    def run(self):
        """启动训练"""
        try:      
            training_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,)
            for i, data in enumerate(training_loader):  # 计划训练批次
                loss, entropy = self.policy_update(data, self.epochs)              
                if (i+1) % 100 == 0:
                    logging.info("Train idx {} : {} / {}".format(i, i*self.batch_size, len(self.dataset)))
            self.policy_value_net.save_model(model_file)
        except KeyboardInterrupt:
            logging.info('quit')

if __name__ == '__main__':
    # train
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    training = FiveChessTrain()
    training.run()