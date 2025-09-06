import os, glob, pickle

from time import time
from model import PolicyValueNet, data_dir, data_wait_dir, model_file

import time, datetime

import os, random
import copy, math

import numpy as np
import torch
from status import save_status_file, read_status_file, set_status_total_value

# 添加 cache 反而更慢
# try:
#     import memcache
#     cache = memcache.Client(['172.17.0.1:11211'], debug=0)
#     print("active memcache cache")
# except:
#     cache = None

# 定义游戏的动作
KEY_NONE, KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN = 0, 1, 2, 3, 4
ACTIONS = [KEY_NONE, KEY_ROTATION, KEY_LEFT, KEY_RIGHT, KEY_DOWN]
ACTIONS_NAME = ["N","O","L","R","D"]
GAME_ACTIONS_NUM = len(ACTIONS) 
GAME_WIDTH, GAME_HEIGHT = 10, 20


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_keep_size, test_size, epochs=5):
        # 训练数据存放路径
        self.data_dir = data_dir                
        # 训练数据最大保存个数
        self.max_keep_size = max_keep_size
        self.test_size = test_size
        self.epoch = epochs
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
        model_prob = torch.from_numpy(data["model_prob"]).float() 
        value = torch.as_tensor(data["value"]).float()
        adv = torch.as_tensor(data["adv"]).float()
        action = torch.as_tensor(data["action"]).long()        
        return state, mcts_prob, model_prob, value, adv, action

    def load_game_files(self):
        print("start load files name ... ")
        start_time = time.time()
        files = glob.glob(os.path.join(self.data_dir, "*.pkl"))
        files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)

        modified_time = os.path.getmtime(files[-1])
        self.first_time = time.localtime(modified_time)
        print("first time:",time.strftime('%y-%m-%d %H:%M:%S', self.first_time))
        modified_time = os.path.getmtime(files[0])
        self.last_time = time.localtime(modified_time)
        print("last time:",time.strftime('%y-%m-%d %H:%M:%S', self.last_time))
        
        delcount = 0
        for i, filename in enumerate(files):
            if i >= self.max_keep_size or os.path.getsize(filename) == 0:
                os.remove(filename)
                # if cache: cache.delete(filename, noreply=True)                     
                delcount += 1
            else:
                self.file_list.append(filename)

        pay_time = round(time.time()-start_time, 2)
        print("loaded data, totle:",len(self.file_list),"delete:", delcount,"paid time:", pay_time)



    def calc_data(self):
        # scores=[]
        print("start load data to memory ...")
        start_time = time.time()
        values={}
        # double_train_list=[]
        for i,fn in enumerate(self.file_list):
            try:
                with open(fn, "rb") as f:
                    state, mcts_prob, model_prob, value, adv, action = pickle.load(f)
                    assert state.shape == (4,20,10) , f'error: sate shape {state.shape}'
                    assert mcts_prob.shape == (5,) , f'error: prob shape {mcts_prob.shape}'
                    assert model_prob.shape == (5,) , f'error: prob shape {model_prob.shape}'
                    assert not np.isnan(value) , f'error: value is Nan'
                    assert not np.isinf(value) , f'error: value is Inf'

                    # if len(mcts_prob)==4:
                    #     mcts_prob = np.concatenate((mcts_prob, np.zeros(1)), axis=0)
            except:
                print("filename {} error can't load".format(fn))
                if os.path.exists(fn): os.remove(fn)
                self.file_list.remove(fn)
                continue
            values[fn]=value

            # 对背景进行 shuffle
            # 保留最近的2行，其余扰动
            # if random.random()>0.9:
            #     board = state[1]
            #     max_idx = np.argmax(board,axis=0)
            #     if not 0 in max_idx:
            #         lines = np.min(20-max_idx)
            #         if lines>2:
            #             lc = random.randint(1,lines-1)
            #             board[lc:]=board[:-lc]
            #             board[:lc]=0                         

            self.data[fn]={"value":value, "state":state, "mcts_prob": mcts_prob, "model_prob": model_prob, "adv": adv, "action": action}
                        
        values_items = list(values.values())
        avg_values = np.average(values_items)
        min_values = np.min(values_items)
        max_values = np.max(values_items)
        std_values = np.std(values_items)

        for fn in values:
            value = values[fn]
            if value>=1: value=1-1e-6
            if value<=-1: value=-1+1e-6
            values[fn] = value


        print("value min/avg/max/std:",[min_values, avg_values, max_values, std_values])        
        self.avg_values = avg_values
        self.std_values = std_values

        for fn in self.data:
            self.data[fn]["value"] = values[fn]
            
        pay_time = round(time.time()-start_time, 2)
        print("loaded to memory, paid time:", pay_time)
        print("load data end")


    def copy_wait_file(self):
        print("start copy wait file to train ...")
        files = glob.glob(os.path.join(data_wait_dir, "*.pkl"))
        movefiles = sorted(files, key=lambda x: os.path.getmtime(x))
        # 等待1秒钟，防止有数据还在写入
        time.sleep(1)
        i = -1
        if len(movefiles)<self.max_keep_size//self.epoch:
            print("SLEEP 60s for watting data, current model file count:",len(movefiles),"need:",self.max_keep_size//5)            
            time.sleep(60)
            raise Exception("NEED SOME NEW DATA TO TRAIN")
        
        for i, fn in enumerate(movefiles):
            filename = os.path.basename(fn)
            savefile = os.path.join(self.data_dir, filename)
            if os.path.exists(savefile): os.remove(savefile)
            os.rename(fn, savefile)
            if self.test_size==-1 or len(self.newsample)<self.test_size:
                self.newsample.append(savefile)
            if (i+1)>=self.max_keep_size//self.epoch and len(movefiles)-i<=self.max_keep_size: break     
            
        # random.shuffle(self.newsample)  
        print("mv %s/%s files to train"%(i+1,len(movefiles)))
            
         
    def curr_size(self):
        return len(self.file_list)      

class Train():
    def __init__(self):
        self.batch_size = 32     # 每批训练的样本，早期用小，防止局部最小值，后期用大，网络平稳 32 64 128 256 512

        # training params
        self.learn_rate = 1e-4
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # MCTS的概率参数，越大越不肯定，训练时1，预测时1e-3
        self.n_playout = 256  # 每个动作的模拟战记录个数
        self.play_batch_size = 1 # 每次自学习次数
        self.buffer_size = 10240  # cache对次数 # 102400 6:30 收集
        self.epochs = 20  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 1e-4  # 策略价值网络KL值目标        
        self.c_puct = 2  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5
   

    def policy_update(self, sample_data, epochs=1):
        """更新策略价值网络policy-value"""
        # 训练策略价值网络
        # 随机抽取data_buffer中的对抗数据
        state_batch, mcts_probs_batch, model_probs_batch, values_batch, advs_batch, actions_batch = sample_data
        # 训练策略价值网络
        p_acc, v_loss, a_loss, p_loss = self.policy_value_net.train_step(state_batch, mcts_probs_batch, model_probs_batch, values_batch, advs_batch, actions_batch, self.learn_rate * self.lr_multiplier)
         
        return p_acc, v_loss, a_loss, p_loss

    def run(self):
        """启动训练"""
        try:
            print("start data loader")
            self.dataset = Dataset(data_dir, self.buffer_size, -1, epochs=self.epochs)
            self.testdataset = copy.copy(self.dataset)
            self.testdataset.test=True
            print("end data loader")

            try:
                self.policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file, l2_const=1e-4)
            except Exception as e:
                print(str(e))
                time.sleep(60)
                return
            self.policy_value_net.save_model(model_file+".bak")           

            dataset_len = len(self.dataset)  
            training_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            testing_loader = torch.utils.data.DataLoader(self.testdataset, batch_size=self.batch_size, shuffle=False,num_workers=0)
            old_probs = None
            test_batch = None

            net = self.policy_value_net.policy_value
            begin_values=None
            begin_act_probs=None
            begin_accuracy=None
            test_data=None
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, test_model_probs, test_values, test_advs, test_action = data
                if i==0:
                    print("test_batch shape:", test_batch.shape, "test_probs shape:", test_probs.shape, 
                          "test_values shape:", test_values.shape, "test_advs shape:", test_advs.shape)
                if test_data==None: test_data=[test_batch, test_probs, test_values]
                test_batch = test_batch.to(self.policy_value_net.device)
                with torch.no_grad(): 
                    act_probs, values = net(test_batch) 
                    if begin_values is None:
                        begin_values = values
                    else:
                        begin_values = np.concatenate((begin_values, values), axis=0)
                    if begin_act_probs is None:
                        begin_act_probs = act_probs
                        begin_accuracy = np.argmax(act_probs, axis=1) == np.argmax(test_probs.cpu().numpy(), axis=1)
                    else:
                        begin_act_probs = np.concatenate((begin_act_probs, act_probs), axis=0)
                        begin_accuracy = np.concatenate((begin_accuracy, np.argmax(act_probs, axis=1)==np.argmax(test_probs.cpu().numpy(), axis=1)), axis=0)
            status = read_status_file()
            self.lr_multiplier = status["total"]["lr_multiplier"]
            self.optimizer_type = status["total"]["optimizer_type"]
            print("lr_multiplier:", self.lr_multiplier, "learn_rate:", self.learn_rate*self.lr_multiplier)

            v_loss_list=[]
            self.policy_value_net.set_optimizer(self.optimizer_type)
            self.policy_value_net.set_learning_rate(self.learn_rate*self.lr_multiplier)
            for i, data in enumerate(training_loader):  # 计划训练批次
                # 使用对抗数据重新训练策略价值网络模型
                p_acc, v_loss, a_loss, p_loss = self.policy_update(data, self.epochs)
                v_loss_list.append(v_loss)
                if i%10 == 0:
                    print(i,"a_loss:", a_loss, "p_loss:", p_loss, "v_loss:", v_loss, "p_acc:", p_acc)
                    # time.sleep(0.1)

                if math.isnan(v_loss) : 
                    print(i,"a_loss:", a_loss, "p_loss:", p_loss, "v_loss:", v_loss, "p_acc:", p_acc)
                    print("v_loss is nan!", )
                    return

            self.policy_value_net.save_model(model_file)
   
            end_values=None
            end_act_probs=None
            end_accuracy=None
            net = self.policy_value_net.policy_value
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, test_model_probs, test_values, test_advs, test_action = data
                test_batch = test_batch.to(self.policy_value_net.device)
                with torch.no_grad(): 
                    act_probs, values = net(test_batch)                 
                    if end_values is None:
                        end_values=values
                    else:
                        end_values=np.concatenate((end_values, values), axis=0)                    
                    if end_act_probs is None:
                        end_act_probs=act_probs
                        end_accuracy=np.argmax(act_probs, axis=1)==np.argmax(test_probs.cpu().numpy(), axis=1)
                    else:
                        end_act_probs=np.concatenate((end_act_probs, act_probs), axis=0)
                        end_accuracy=np.concatenate((end_accuracy, np.argmax(act_probs, axis=1)==np.argmax(test_probs.cpu().numpy(), axis=1)), axis=0)

            begin_act_probs_e = np.exp(begin_act_probs-np.max(begin_act_probs, axis=1, keepdims=True))
            begin_act_probs = begin_act_probs_e/np.sum(begin_act_probs_e, axis=1, keepdims=True)
            end_act_probs_e = np.exp(end_act_probs-np.max(end_act_probs, axis=1, keepdims=True))
            end_act_probs = end_act_probs_e/np.sum(end_act_probs_e, axis=1, keepdims=True)
            
            # for i in range(len(begin_values)):
                # print("value[{}] begin:{} end:{} to:{}".format(i, begin_values[i], end_values[i], test_data[2][i].numpy()))  
                # if i>=4:break
            print("目标：")
            print(test_data[2].numpy()[:24])    
            print("初始：")
            print(begin_values[:24])
            print("结束：")
            print(end_values[:24])
            for i in range(len(begin_values)):
                idx = np.argmax(begin_act_probs[i])
                print("probs[{}] begin:{} end:{} to:{} ".format(i, begin_act_probs[i][idx], end_act_probs[i][idx], test_data[1][i].numpy()[idx]))
                if i>=4:break

            print("probs begin_accuracy:", np.mean(begin_accuracy), "end_accuracy:", np.mean(end_accuracy))
            # kl = np.mean(np.sum(begin_act_probs * (np.log(begin_act_probs) - np.log(end_act_probs)), axis=1))
            kl = np.mean(np.sum(begin_act_probs * (np.log(begin_act_probs/end_act_probs)), axis=1))
            if np.isnan(kl):
                kl = 0
                
            status = read_status_file()
            set_status_total_value(status, "kl", kl, 0.1)
            total_kl = status["total"]["kl"]

            if total_kl > self.kl_targ*2 :
                self.lr_multiplier /= 1.1
            elif total_kl < self.kl_targ/2 :
                self.lr_multiplier *= 1.1
            if self.lr_multiplier < 0.1:  self.lr_multiplier = 0.1
            if self.lr_multiplier > 10:  self.lr_multiplier = 10
            
            status["total"]["lr_multiplier"] = float(self.lr_multiplier) 
            save_status_file(status)    

            print("kl:{} vs {} lr_multiplier:{} lr:{} avg_values:{} std_values:{}".format(kl, self.kl_targ, self.lr_multiplier, self.learn_rate*self.lr_multiplier, \
                self.dataset.avg_values, self.dataset.std_values))
            
        except KeyboardInterrupt:
            print('quit')

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print('start training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    training = Train()
    training.run()
    print('end training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

