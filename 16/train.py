import os, glob, pickle

from time import time
from model import PolicyValueNet, data_dir, data_wait_dir, model_file

import time, datetime

import os, random
import copy, math

import numpy as np
import torch

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

        modified_time = os.path.getmtime(files[-1])
        self.first_time = time.localtime(modified_time)
        print("first time:",time.strftime('%y-%m-%d %H:%M:%S', self.first_time))
        modified_time = os.path.getmtime(files[0])
        self.last_time = time.localtime(modified_time)
        print("last time:",time.strftime('%y-%m-%d %H:%M:%S', self.last_time))
        
        delcount = 0
        for i, filename in enumerate(files):
            if i >= self.max_keep_size:
                os.remove(filename)
                # if cache: cache.delete(filename, noreply=True)                     
                delcount += 1
            else:
                self.file_list.append(filename)

        # random.shuffle(self.file_list)
        pay_time = round(time.time()-start_time, 2)
        print("loaded data, totle:",len(self.file_list),"delete:", delcount,"paid time:", pay_time)
        if len(self.file_list)<self.max_keep_size/100 :
            print("SLEEP 60s for %s to %s data."%(len(self.file_list), self.max_keep_size/100))
            time.sleep(60)
            raise Exception("NEED SOME NEW DATA TO TRAIN")


    def calc_data(self):
        # scores=[]
        print("start load data to memory ...")
        start_time = time.time()
        scores={}
        values={}
        # double_train_list=[]
        for fn in self.file_list:
            try:
                # if cache: 
                #     obj = cache.get(fn)
                #     if obj!=None:
                #         state, mcts_prob, value, score = obj
                #     else:   
                #         with open(fn, "rb") as f:
                #             state, mcts_prob, value, score = pickle.load(f)
                #             cache.add(fn, (state, mcts_prob, value, score))
                # else:
                with open(fn, "rb") as f:
                    state, mcts_prob, value, score = pickle.load(f)
                    # print(state[0])
                    # print(state[1])
            except:
                print("filename {} error can't load".format(fn))
                if os.path.exists(fn): os.remove(fn)
                self.file_list.remove(fn)
                continue
            scores[fn]=score
            values[fn]=value

            # b,h,w = state.shape
            # for j in range(h):
            #     if random.random()>0.5: continue
            #     if np.min(state[1][j]) < 0: continue
            #     c = np.sum(state[1][j]) 
            #     if c==0 or c==1 or c==w-1: continue
            #     idx = random.randint(0,w-1)
            #     if state[1][j][idx]==-1: continue
            #     v = 0 if state[1][j][idx]==1 else 1            
            #     for i in range(1,b):
            #          state[i][j][idx]=v

            # if score%1==0:
            #     self.data[fn]={"value":-1/(score+1), "state":state, "mcts_prob": mcts_prob}
            # else:
            # self.data[fn]={"value":-1/(piececount**0.5), "state":state, "mcts_prob": mcts_prob}

            # 未来的收益，评估当前局面的状态，但这个收益有点扩大了
            self.data[fn]={"value":0, "state":state, "mcts_prob": mcts_prob}
        values_items = list(values.values())
        avg_values = np.average(values_items)
        min_values = np.min(values_items)
        max_values = np.max(values_items)
        std_values = np.std(values_items)

        print("value min/avg/max/std:",[min_values, avg_values, max_values, std_values])        

        scores_items = list(scores.values())
        avg_scores = np.average(scores_items)
        std_scores = np.std(scores_items)
        min_scores = np.min(scores_items)
        max_scores = np.max(scores_items)
        print("score min/avg/max/std:",[min_scores, avg_scores, max_scores, std_scores])
        for fn in self.data:
            # self.data[fn]["value"] = (values[fn] - avg_values)/(max_values-min_values) + (scores[fn]-avg_scores)/(max_scores-min_scores)            
            # self.data[fn]["value"] = (values[fn]+scores[fn])*0.5 - 1 
            # self.data[fn]["value"] = (scores[fn]-min_scores)*2/(max_scores-min_scores) - 1
            self.data[fn]["value"] = values[fn]/2
            # self.data[fn]["value"] = scores[fn]

        pay_time = round(time.time()-start_time, 2)
        print("loaded to memory, paid time:", pay_time)
        # print("piececount avg:", avg_piececount, "var:", var_piececount,"min:", min_piececount,"max:", max_piececount)
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
            if (i>=49 and i>len(movefiles)*0.05) or i>=self.max_keep_size//2: break       
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
        self.batch_size = 256     # 每批训练的样本，早期用小，防止局部最小值，后期用大，网络平稳 32 64 128 256 512

        # training params
        self.learn_rate = 5e-5
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # MCTS的概率参数，越大越不肯定，训练时1，预测时1e-3
        self.n_playout = 256  # 每个动作的模拟战记录个数
        self.play_batch_size = 1 # 每次自学习次数
        self.buffer_size = 80000  # cache对次数
        self.epochs = 1  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标        
        self.c_puct = 2  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5
   

    def policy_update(self, sample_data, epochs=1):
        """更新策略价值网络policy-value"""
        # 训练策略价值网络
        # 随机抽取data_buffer中的对抗数据
        state_batch, mcts_probs_batch, values_batch = sample_data
        # 训练策略价值网络
        # for i in range(epochs):
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

            try:
                self.policy_value_net = PolicyValueNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file)
            except Exception as e:
                print(str(e))
                time.sleep(60)
                # os.rename(model_file, model_file+"_err")
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
            test_data=None
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, test_values = data
                if test_data==None: test_data=[test_batch, test_probs, test_values]
                test_batch = test_batch.to(self.policy_value_net.device)
                # test_values = test_values.to(self.policy_value_net.device)
                with torch.no_grad(): 
                    act_probs, values = net(test_batch) 
                    # print("value[0] dst:{} pred_s:{}".format(test_values[:5].cpu().numpy(), values[:5]))  
                    # print("probs[0] dst:{} pred_s:{}".format(test_probs[0].cpu().numpy(), act_probs[0]))
                    if begin_values is None:
                        begin_values = values
                    else:
                        begin_values = np.concatenate((begin_values, values), axis=0)
                    if begin_act_probs is None:
                        begin_act_probs = act_probs
                    else:
                        begin_act_probs = np.concatenate((begin_act_probs, act_probs), axis=0)
                    # begin_values.append(values[:5])
                    # begin_act_probs.append(act_probs[0])
            self.train_conf = {"lr_multiplier":1,"optimizer_type":0}
            train_conf_file=os.path.join(data_dir,"train_conf_pkl")
            if os.path.exists(train_conf_file):
                with open(train_conf_file, "rb") as fn:
                    self.train_conf = pickle.load(fn)
                    self.lr_multiplier = self.train_conf["lr_multiplier"]
                    print("lr_multiplier:", self.lr_multiplier, "learn_rate:", self.learn_rate*self.lr_multiplier)
            v_loss_list=[]
            self.policy_value_net.set_optimizer(self.train_conf["optimizer_type"])
            self.policy_value_net.set_learning_rate(self.learn_rate*self.lr_multiplier)
            for i, data in enumerate(training_loader):  # 计划训练批次
                # 使用对抗数据重新训练策略价值网络模型
                _, v_loss, p_loss = self.policy_update(data, self.epochs)
                v_loss_list.append(v_loss)
                if i%10 == 0:
                    print(i,"v_loss:",v_loss,"p_loss",p_loss)
                    time.sleep(0.1)

                if math.isnan(v_loss): 
                    print("v_loss is nan!")
                    return

                # if i%10 == 0:
                #     print(("TRAIN idx {} : {} / {} v_loss:{:.5f}, p_loss:{:.5f}")\
                #         .format(i, i*self.batch_size, dataset_len, v_loss, p_loss))

                #     # 动态调整学习率
                #     if old_probs is None:
                #         test_batch, test_probs, test_values = data
                #         old_probs, old_value = self.policy_value_net.policy_value(test_batch) 
                #     else:
                #         new_probs, new_value = self.policy_value_net.policy_value(test_batch)
                #         kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
                        
                #         old_probs = None
                        
                #         if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                #             self.lr_multiplier /= 1.5
                #         elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                #             self.lr_multiplier *= 1.5
                #         else:
                #             continue
                #         print("kl:{} vs {} lr_multiplier:{} lr:{}".format(kl, self.kl_targ, self.lr_multiplier, self.learn_rate*self.lr_multiplier))

            self.policy_value_net.save_model(model_file)
   
            end_values=None
            end_act_probs=None
            net = self.policy_value_net.policy_value
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, test_values = data
                test_batch = test_batch.to(self.policy_value_net.device)
                with torch.no_grad(): 
                    act_probs, values = net(test_batch) 
                    if end_values is None:
                        end_values=values
                    else:
                        end_values=np.concatenate((end_values, values), axis=0)
                    if end_act_probs is None:
                        end_act_probs=act_probs
                    else:
                        end_act_probs=np.concatenate((end_act_probs, act_probs), axis=0)

            for i in range(len(begin_values)):
                print("value[{}] begin:{} end:{} to:{}".format(i, begin_values[i], end_values[i], test_data[2][i].numpy()))  
                if i>=4:break
            for i in range(len(begin_values)):
                print("probs[{}] begin:{} end:{} to:{} ".format(i, begin_act_probs[i], end_act_probs[i],test_data[1][i].numpy()))
                if i>=4:break
                
            kl = np.mean(np.sum(begin_act_probs * (np.log(begin_act_probs + 1e-10) - np.log(end_act_probs + 1e-10)), axis=1))
            print("act_probs, kl:",kl)
            if kl > self.kl_targ * 2 and self.lr_multiplier > 0.01:
                self.lr_multiplier /= 1.5
            elif kl < self.kl_targ / 2 and self.lr_multiplier < 100:
                self.lr_multiplier *= 1.5
            if self.learn_rate*self.lr_multiplier>1e-3: self.lr_multiplier = 1e-3/self.learn_rate
            if self.learn_rate*self.lr_multiplier<1e-6: self.lr_multiplier = 1e-6/self.learn_rate
            print("kl:{} vs {} lr_multiplier:{} lr:{}".format(kl, self.kl_targ, self.lr_multiplier, self.learn_rate*self.lr_multiplier))
            with open(train_conf_file, 'wb') as fn:
                self.train_conf["lr_multiplier"] = self.lr_multiplier
                # if self.train_conf["optimizer_type"]==0 and np.average(v_loss_list)<0.1:
                #     self.train_conf["optimizer_type"]=1
                #     self.train_conf["lr_multiplier"]=1
                # if self.train_conf["optimizer_type"]==1 and np.average(v_loss_list)>0.2:
                #     self.train_conf["optimizer_type"]=0
                #     self.train_conf["lr_multiplier"]=1
                pickle.dump(self.train_conf, fn)


        except KeyboardInterrupt:
            print('quit')

if __name__ == '__main__':
    # train
    # logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print('start training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    training = Train()
    training.run()
    print('end training',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

