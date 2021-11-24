import os, glob, pickle

from time import time
from model import PolicyValueNet
from agent import Agent, ACTIONS

import sys, time

from itertools import count
from collections import deque
from collections import namedtuple
import os, math, random

from threading import Thread, Lock

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
model_file =  os.path.join(model_dir, 'model-resnet.pth')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_keep_size):
        # 训练数据存放路径
        self.data_dir = data_dir                
        # 训练数据最大保存个数
        self.max_keep_size = max_keep_size
        # 当前训练数据索引保存文件
        self.data_index_file = os.path.join(data_dir, 'index.txt')
        self.file_list = deque(maxlen=max_keep_size)    
        self.newsample = []
        self.load_index()
        self.copy_wait_file()
        self.load_game_files()
        self.sample=0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        # 状态，步骤的概率，最终得分
        while True:
            try:
                state, qvals = pickle.load(open(filename, "rb"))
                break
            except:
                print("filename {} error can't load".format(filename))
                os.remove(filename)
                filename = random.choice(self.file_list)
           
        state = torch.from_numpy(state).float()
        qvals = torch.as_tensor(qvals).float()
        return state, qvals

    def load_game_files(self):
        files = glob.glob(os.path.join(self.data_dir, "*.pkl"))
        files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
        for i,filename in enumerate(files):
            if i >= self.max_keep_size:
                os.remove(filename)
                print("delete", filename)
            else:
                self.file_list.append(filename)

    def save_index(self):
        with open(self.data_index_file, "w") as f:
            f.write(str(self.index))

    def load_index(self):
        if os.path.exists(self.data_index_file):
            self.index = int(open(self.data_index_file, 'r').read().strip())
        else:
            self.index = 0

    def copy_wait_file(self):
        movefiles=os.listdir(data_wait_dir)
        # 等待5秒钟，防止有数据还在写入
        time.sleep(5)
        i = -1
        for i, fn in enumerate(movefiles):
            filename = "{}.pkl".format(self.index % self.max_keep_size,)
            savefile = os.path.join(self.data_dir, filename)
            if os.path.exists(savefile): os.remove(savefile)
            os.rename(os.path.join(data_wait_dir,fn), savefile)
            self.index += 1
            self.save_index() 
            self.newsample.append(savefile)
            if i>=100 and i>len(movefiles)*0.1: break       
        print("mv %s/%s files to train"%(i,len(movefiles)))
        if i==-1:
            print("SLEEP 60s for watting data")
            time.sleep(60)
            raise Exception("NEED SOME NEW DATA TO TRAIN")

    # 保存新的训练样本，但不参与到本次训练，等下一次训练加载
    def save(self, obj):
        # 文件名为buffer取余，循环保存
        filename = "{}.pkl".format(self.curr_game_batch_num % self.buffer_size,)
        savefile = os.path.join(self.data_dir, filename)
        pickle.dump(obj, open(savefile, "wb"))
        self.curr_game_batch_num += 1
        self.save_game_batch_num()
        
    def curr_size(self):
        return len(self.file_list)

class TestDataset(Dataset):
    def __init__(self, data_dir, max_keep_size, file_list):
        # 训练数据存放路径
        self.data_dir = data_dir                
        # 训练数据最大保存个数
        self.max_keep_size = max_keep_size
        # 当前训练数据索引保存文件
        self.data_index_file = os.path.join(data_dir, 'index.txt')
        self.file_list = deque(maxlen=max_keep_size) 
        for file in file_list:
            self.file_list.append(file) 
        self.sample=0  
       

class Train():
    def __init__(self):
        self.game_batch_num = 2000000  # selfplay对战次数
        self.batch_size = 512     # data_buffer中对战次数超过n次后开始启动模型训练

        # training params
        self.learn_rate = 1e-5
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # MCTS的概率参数，越大越不肯定，训练时1，预测时1e-3
        self.n_playout = 64  # 每个动作的模拟战记录个数
        self.play_batch_size = 1 # 每次自学习次数
        self.buffer_size = 200000  # cache对次数
        self.epochs = 1  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        self.c_puct = 0.1  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5
   

    def policy_update(self, sample_data, epochs=1):
        """更新策略价值网络policy-value"""
        # 训练策略价值网络
        # 随机抽取data_buffer中的对抗数据
        state_batch, qval_batch = sample_data
        # 训练策略价值网络
        for i in range(epochs):
            loss, v_loss, p_loss, entropy = self.policy_value_net.train_step(state_batch, qval_batch, self.learn_rate * self.lr_multiplier)
         
        return loss, v_loss, p_loss, entropy

    def run(self):
        """启动训练"""
        try:
            print("start data loader")
            self.dataset = Dataset(data_dir, self.buffer_size)
            newsample=self.dataset.newsample
            self.testdataset = TestDataset(data_dir, 10, newsample)
            print("end data loader")

            self.policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)
            self.policy_value_net.save_model(model_file+".bak")
            
            
            dataset_len = len(self.dataset)  
            training_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=1,)
            testing_loader = torch.utils.data.DataLoader(self.testdataset, batch_size=self.batch_size, shuffle=True, num_workers=1,)
            old_probs = None
            test_batch = None
            for i, data in enumerate(training_loader):  # 计划训练批次
                if i==0:
                    _batch, _qvals = data
                    print(_batch[0][0])
                    print(_batch[0][1])
                    print(_qvals[0])

                # 使用对抗数据重新训练策略价值网络模型
                _, v_loss, p_loss, entropy = self.policy_update(data, self.epochs)
                if i%10 == 0:
                    print(("TRAIN idx {} : {} / {} v_loss:{:.5f}, p_loss:{:.5f}, entropy:{:.5f}")\
                        .format(i, i*self.batch_size, dataset_len, v_loss, p_loss, entropy))

                    # 动态调整学习率
                    if old_probs is None:
                        test_batch, test_qvals = next(iter(testing_loader))
                        old_probs, _, old_value = self.policy_value_net.policy_value(test_batch) 
                    else:
                        new_probs, _, new_value = self.policy_value_net.policy_value(test_batch)
                        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
                        
                        if i % 50 == 0:   
                            print("probs[0] old:{}".format(old_probs[0]))   
                            print("probs[0] new:{}".format(new_probs[0]))   
                            print("probs[0] tg: {}".format(test_probs[0])) 
                            maxlen = min(10, len(test_value)) 
                            for j in range(maxlen): 
                                print("value[0] old:{} new:{} tg:{}".format(old_value[j][0], new_value[j][0], test_value[j]))  

                        old_probs = None
                        
                        if kl > self.kl_targ * 2:
                            self.lr_multiplier /= 1.5
                        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                            self.lr_multiplier *= 1.5
                        else:
                            continue
                        print("kl:{} lr_multiplier:{} lr:{}".format(kl, self.lr_multiplier, self.learn_rate*self.lr_multiplier))

            self.policy_value_net.save_model(model_file)
   
    
        except KeyboardInterrupt:
            print('quit')

if __name__ == '__main__':
    # train
    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    training = Train()
    training.run()

