import os, glob, pickle

from time import time
from model import PolicyValueNet, data_dir, data_wait_dir, model_file

import time, datetime

import os, random
import copy, math

import numpy as np
import torch

# 定义游戏的动作
KEY_NONE, KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN = 0, 1, 2, 3, 4
ACTIONS = [KEY_NONE, KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
ACTIONS_NAME = ["N","O","L","R","D"]
GAME_ACTIONS_NUM = len(ACTIONS) 
GAME_WIDTH, GAME_HEIGHT = 10, 20


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_keep_size, test_size):
        # 训练数据存放路径
        self.data_dir = data_dir                
        # 训练数据最大保存个数
        self.max_keep_size = max_keep_size
        self.test_size = test_size
        self.file_list = [] # deque(maxlen=max_keep_size)    
        self.newsample = []
        self.data={}
        self.copy_wait_file()
        self.load_game_files()
        self.calc_data()
        self.test=False


    def __len__(self):
        if self.test:
            return len(self.newsample)
        else:
            return len(self.file_list)

    def __getitem__(self, index):
        if self.test:
            fn = self.newsample[index]
        else:
            fn = self.file_list[index]

        data = self.data[fn]
            
        # 状态，步骤的概率，最终得分
        state = torch.from_numpy(data["state"]).float()
        mcts_prob = torch.from_numpy(data["mcts_prob"]).float()
        value = torch.as_tensor(data["value"]).float()
        return state, mcts_prob, value

    def load_game_files(self):
        print("start load files name ... ")
        start_time = time.time()
        files = glob.glob(os.path.join(self.data_dir, "*.pkl"))
        files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)

        modified_time = os.path.getmtime(files[0])
        convert_time = time.localtime(modified_time)
        print("first time:",time.strftime('%Y-%m-%d %H:%M:%S', convert_time))
        modified_time = os.path.getmtime(files[-1])
        convert_time = time.localtime(modified_time)
        print("last time:",time.strftime('%Y-%m-%d %H:%M:%S', convert_time))
        
        delcount = 0
        for i, filename in enumerate(files):
            if i >= self.max_keep_size:
                os.remove(filename)
                delcount += 1
            else:
                self.file_list.append(filename)

        # random.shuffle(self.file_list)
        pay_time = round(time.time()-start_time, 2)
        print("loaded data, totle:",len(self.file_list),"delete:", delcount,"paid time:", pay_time)
        if len(self.file_list)<self.max_keep_size/100 :
            print("SLEEP 60s for %s to %s data."%(len(self.file_list), self.max_keep_size/10))
            time.sleep(60)
            raise Exception("NEED SOME NEW DATA TO TRAIN")


    def calc_data(self):
        # scores=[]
        print("start load data to memory ...")
        start_time = time.time()
        piececounts=[]
        # double_train_list=[]
        for fn in self.file_list:
            try:
                with open(fn, "rb") as f:
                    state, mcts_prob, piececount, qval = pickle.load(f)
                    assert state.shape[0] == 8                         
                    # if abs(qval)>0.5:
                    #     double_train_list.append(fn)
            except:
                print("filename {} error can't load".format(fn))
                if os.path.exists(fn): os.remove(fn)
                self.file_list.remove(fn)
                continue
            piececounts.append(piececount)
            
            s,h,w = state.shape

            canzero=[]
            for i in range(w):
                check=False
                for j in range(10,20):
                    if state[1][j][i]==1:
                        if not check:
                            check=True
                        else:
                            canzero.append([j,i])
            canzero_len = len(canzero)
            if canzero_len>10:
                zerolist = random.sample(canzero,random.randint(1,canzero_len//2))
                for zero in zerolist:
                    for i in range(s):
                        state[i][zero[0]][zero[1]]=0
            self.data[fn]={"value":piececount, "state":state, "mcts_prob": mcts_prob}

        avg_piececount = np.average(piececounts)
        var_piececount = np.var(piececounts)
        min_piececount = np.min(piececounts)
        max_piececount = np.max(piececounts)
        per_piececount = np.percentile(piececounts,(25,50,75), method="midpoint")
        print("midpoint(25%,50%,75%):",per_piececount)
        print("var/avg/min/max:",[var_piececount,avg_piececount,min_piececount,max_piececount])
        dif_piececount = per_piececount[2]-per_piececount[0]
        avg_piececount = per_piececount[1]
        if var_piececount<1:
            print("SLEEP 60s for piececount var: %s avg: %s data."%(var_piececount, avg_piececount))
            time.sleep(60)
            raise Exception("NEED SOME NEW DATA TO TRAIN")

        for fn in self.data:
            self.data[fn]["value"]=(self.data[fn]["value"]-avg_piececount)/dif_piececount

        # 将qval高的重复学习一次    
        # self.file_list.extend(double_train_list)

        pay_time = round(time.time()-start_time, 2)
        print("loaded to memory, paid time:", pay_time)
        print("piececount avg:", avg_piececount, "var:", var_piececount,"min:", min_piececount,"max:", max_piececount)
        print("load data end")


    def copy_wait_file(self):
        print("start copy wait file to train ...")
        files = glob.glob(os.path.join(data_wait_dir, "*.pkl"))
        movefiles = sorted(files, key=lambda x: os.path.getmtime(x))
        # 等待1秒钟，防止有数据还在写入
        time.sleep(1)
        i = -1
        for i, fn in enumerate(movefiles):
            filename = os.path.basename(fn)
            savefile = os.path.join(self.data_dir, filename)
            if os.path.exists(savefile): os.remove(savefile)
            os.rename(fn, savefile)
            if len(self.newsample)<self.test_size:
                self.newsample.append(savefile)
            if (i>=99 and i>len(movefiles)*0.01) or i>=self.max_keep_size//2: break       
            if i>=self.max_keep_size//10: break       
        print("mv %s/%s files to train"%(i+1,len(movefiles)))
        if i==-1 :
            print("SLEEP 60s for watting data")
            time.sleep(60)
            raise Exception("NEED SOME NEW DATA TO TRAIN")
         
    def curr_size(self):
        return len(self.file_list)      

class Train():
    def __init__(self):
        self.game_batch_num = 2000000  # selfplay对战次数
        self.batch_size = 512     # 每批训练的样本，早期用小，防止局部最小值，后期用大，网络平稳 32 64 128 256 512

        # training params
        self.learn_rate = 1e-4
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # MCTS的概率参数，越大越不肯定，训练时1，预测时1e-3
        self.n_playout = 128  # 每个动作的模拟战记录个数
        self.play_batch_size = 1 # 每次自学习次数
        self.buffer_size = 320000  # cache对次数
        self.epochs = 1  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标        
        self.c_puct = 2  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5
   

    def policy_update(self, sample_data, epochs=1):
        """更新策略价值网络policy-value"""
        # 训练策略价值网络
        # 随机抽取data_buffer中的对抗数据
        state_batch, mcts_probs_batch, values_batch = sample_data
        # 训练策略价值网络
        for i in range(epochs):
            loss, v_loss, p_loss= self.policy_value_net.train_step(state_batch, mcts_probs_batch, values_batch, self.learn_rate * self.lr_multiplier)
         
        return loss, v_loss, p_loss

    def run(self):
        """启动训练"""
        try:
            print("start data loader")
            self.dataset = Dataset(data_dir, self.buffer_size, self.batch_size*5)
            self.testdataset = copy.copy(self.dataset)
            self.testdataset.test=True
            print("end data loader")

            self.policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)
            self.policy_value_net.save_model(model_file+".bak")           

            dataset_len = len(self.dataset)  
            training_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            testing_loader = torch.utils.data.DataLoader(self.testdataset, batch_size=self.batch_size, shuffle=False,num_workers=0)
            old_probs = None
            test_batch = None

            loss_fn = torch.nn.MSELoss()
            net = self.policy_value_net.policy_value
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, test_values = data
                test_batch = test_batch.to(self.policy_value_net.device)
                test_values = test_values.to(self.policy_value_net.device)
                with torch.no_grad(): 
                    act_probs, values = net(test_batch) 
                    print("value[0] to:{} 0:{}".format(test_values[:5].cpu().numpy(), values[:5]))  
                    print("probs[0] to:{} 0:{}".format(test_probs[0].cpu().numpy(), act_probs[0]))

            for i, data in enumerate(training_loader):  # 计划训练批次
                # 使用对抗数据重新训练策略价值网络模型
                _, v_loss, p_loss = self.policy_update(data, self.epochs)
                if i%10 == 0:
                    print(("TRAIN idx {} : {} / {} v_loss:{:.5f}, p_loss:{:.5f}")\
                        .format(i, i*self.batch_size, dataset_len, v_loss, p_loss))

                    # 动态调整学习率
                    if old_probs is None:
                        test_batch, test_probs, test_values = data
                        old_probs, old_value = self.policy_value_net.policy_value(test_batch) 
                    else:
                        new_probs, new_value = self.policy_value_net.policy_value(test_batch)
                        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
                        
                        old_probs = None
                        
                        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                            self.lr_multiplier /= 1.5
                        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                            self.lr_multiplier *= 1.5
                        else:
                            continue
                        print("kl:{} lr_multiplier:{} lr:{}".format(kl, self.lr_multiplier, self.learn_rate*self.lr_multiplier))

            self.policy_value_net.save_model(model_file)
   

            net = self.policy_value_net.policy_value
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, test_values = data
                test_batch = test_batch.to(self.policy_value_net.device)
                test_values = test_values.to(self.policy_value_net.device)
                with torch.no_grad(): 
                    act_probs, values = net(test_batch) 
                    print("value[0] to:{} 1:{}".format(test_values[:5].cpu().numpy(), values[:5]))  
                    print("probs[0] to:{} 1:{}".format(test_probs[0].cpu().numpy(), act_probs[0]))

        except KeyboardInterrupt:
            print('quit')

if __name__ == '__main__':
    # train
    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    print('start training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    training = Train()
    training.run()
    print('end training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

