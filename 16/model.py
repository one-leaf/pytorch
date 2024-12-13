from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from collections import OrderedDict
from torchvision.models import resnet34
from vit import VitNet
import random

# 定义游戏的保存文件名和路径
model_name = "vit-ti" # "vit" # "mlp"
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, 'data', model_name)
if not os.path.exists(data_dir): os.makedirs(data_dir)
data_wait_dir = os.path.join(curr_dir, 'data', model_name, 'wait')
if not os.path.exists(data_wait_dir): os.makedirs(data_wait_dir)
model_dir = os.path.join(curr_dir, 'model', model_name)
if not os.path.exists(model_dir): os.makedirs(model_dir)
model_file =  os.path.join(model_dir, 'model.pth')


class PolicyValueNet():
    def __init__(self, input_width, input_height, output_size, model_file=None, device=None, l2_const=5e-5):
        self.input_channels = 4 # 输入通道数
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_width * input_height
        self.output_size = output_size
        self.input_channels = 4 # 输入通道数
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        print("use", device)

        self.l2_const = l2_const  
        # ViT-Ti : depth 12 width 192 heads 3 LR=4e-3
        self.policy_value_net = VitNet(embed_dim=192, depth=6, num_heads=3, num_classes=5, num_quantiles=128, drop_ratio=0.1, drop_path_ratio=0.1, attn_drop_ratio=0.1, num_channels=self.input_channels)
        # ViT-S : depth 12 width 386 heads 6
        # self.policy_value_net = VitNet(embed_dim=386, depth=12, num_heads=6, num_classes=4, num_quantiles=128)
        # ViT-B : depth 12 width 768 heads 12
        # self.policy_value_net = VitNet(num_classes=4, num_quantiles=12)
        self.policy_value_net.to(device)
        # self.print_netwark()

        # 前期用这个
        # self.optimizer = optim.AdamW(self.policy_value_net.parameters(), lr=1e-6, weight_decay=self.l2_const)  
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=1e-6, weight_decay=self.l2_const)  
        # transformers use adam 
        # self.optimizer = optim.SGD(self.policy_value_net.parameters(), lr=1e-6, momentum=0.9, weight_decay=self.l2_const)

        # 测试失败，学不会
        # self.optimizer = Lion(self.policy_value_net.parameters(), lr=1e-6, weight_decay=self.l2_const)

        if model_file and os.path.exists(model_file):
            print("Loading model", model_file)
            net_sd = torch.load(model_file, map_location=self.device)
            # try:
            self.policy_value_net.load_state_dict(net_sd, strict=False)
            # except:
            #     # net_sd = {k: v for k, v in net_sd.items() if k in net_sd and 'act_dist' not in k}
            #     print(net_sd["act_dist.weight"].shape)
            #     print(net_sd["act_dist.bias"].shape)
                
            #     zero_col,_ = torch.min(net_sd["act_dist.weight"],dim=0)
            #     zero_col=zero_col.view(1,-1)
            #     print(zero_col.shape)
            #     net_sd["act_dist.weight"]=torch.cat((net_sd["act_dist.weight"],zero_col))
                
            #     zero_col,_ = torch.min(net_sd["act_dist.bias"],dim=0)
            #     zero_col=zero_col.view(1)
            #     net_sd["act_dist.bias"]=torch.cat((net_sd["act_dist.bias"],zero_col))
                
            #     print(net_sd["act_dist.weight"].shape)
            #     print(net_sd["act_dist.bias"].shape)
                
            #     model_dict = self.policy_value_net.state_dict()
            #     model_dict.update(net_sd)
            #     self.policy_value_net.load_state_dict(model_dict, strict=False)
        else:
            self.save_model(model_file)
        self.lr = 0
        self.cache = {}

    def set_optimizer(self, type_id):
        if type_id==0:
            # self.optimizer = Lion(self.policy_value_net.parameters(), lr=1e-5, weight_decay=self.l2_const)
            self.optimizer = optim.AdamW(self.policy_value_net.parameters(), lr=1e-5, weight_decay=self.l2_const) 
        else:
            self.optimizer = optim.SGD(self.policy_value_net.parameters(), lr=1e-5, momentum=0.9, weight_decay=self.l2_const)

    # 设置学习率
    def set_learning_rate(self, lr):
        if self.lr != lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
            print("Set modle learn rate to:", lr)

    # 打印当前网络
    def print_netwark(self):
        x = torch.Tensor(1,4,20,10).to(self.device)
        print(self.policy_value_net)
        v, p = self.policy_value_net(x)
        print("value:", v.size())
        print("policy:", p.size())

    # 根据当前状态得到，action的概率和概率
    def policy_value(self, state_batch):
        """
        输入: 一组游戏的当前状态
        输出: 一组动作的概率和动作的得分
        """
        if torch.is_tensor(state_batch):
            state_batch_tensor = state_batch.to(self.device)
        else:
            state_batch_tensor = torch.FloatTensor(state_batch).to(self.device)

        self.policy_value_net.eval()
        with torch.no_grad(): 
            act_probs, value, reward = self.policy_value_net.forward(state_batch_tensor)    #[b, num_classes] [b, num_quantiles]
            
        # act_probs = torch.softmax(act_probs,dim=1)
        # num_quantiles = value.shape[1]
        
        # alphatensor 0.75 取尾端，偏向负值用于探索
        # num_value =  int(num_quantiles * 0.75)
        # value =  torch.mean(value[:, num_value:] , dim=1)
        # 真实应该取均值        
        # value =  torch.mean(value, dim=1)
        
        # 这边按0.75取        
        # num_value =  int(num_quantiles * 0.75)
        # value =  torch.mean(value[:, num_value:] , dim=1)
        value =  torch.mean(value, dim=1)

        # 还原成标准的概率
        act_probs = act_probs.cpu().numpy()
        value = value.cpu().numpy()
        reward = reward.cpu().numpy()
        return act_probs, value, reward

    # 从当前游戏获得 ((action, act_probs),...) 的可用动作+概率和当前游戏胜率
    def policy_value_fn(self, game, only_Cache_Next=False):
        """
        输入: 游戏
        输出: 一组（动作， 概率）和游戏当前状态的胜率
        """     
        if only_Cache_Next:
            k_list = []
            s_list = []
            while not game.terminal:
                nz_idx = np.nonzero(game.availables)[0]
                game.step(random.choice(nz_idx))
                key = game.key
                if key not in self.cache:
                    k_list.append(key)
                    s_list.append(game.current_state().copy())
                    if len(k_list) == 8:
                        break
            if len(k_list) > 0:
                current_state = np.array(s_list)
                act_probs, value, reward = self.policy_value(current_state)
                for i in range(len(k_list)):
                    self.cache[k_list[i]] = (act_probs[i], value[i], reward[i,0])
            return None, None, None
        
        key = game.key
        if key in self.cache:
            act_probs, value, reward = self.cache[key]
            return act_probs, value, reward
        
        current_state = np.array([game.current_state()])#game.current_state().reshape(1, self.input_channels, self.input_height, self.input_width)
        act_probs, value, reward = self.policy_value(current_state)
        act_probs=act_probs[0]
        value=value[0]
        reward = reward[0, 0]
        
        self.cache[key] = (act_probs, value, reward)
        return act_probs, value, reward
    
    def policy_value_fn_best_act(self, game):
        """
        输入: 游戏
        输出: 一组（动作， 概率）和游戏当前状态的胜率
        """  
        act_probs,_,_ = self.policy_value_fn(game)
        
        actions = game.availables
        idx = np.argmax(act_probs*actions)
        return idx

    # 价值网络损失
    def quantile_regression_loss(self, quantiles, target):
        num_quantiles = quantiles.shape[1]
        tau = (torch.arange(num_quantiles).to(self.device) + 0.5) / num_quantiles                   #[b, num_quantiles]
        newtarget = target.unsqueeze(1)                                                #[b, 1]
        newtarget = newtarget.repeat(1, num_quantiles)                                    #[b, num_quantiles]
        weights = torch.where(quantiles > newtarget, tau, 1 - tau)                     #[b, num_quantiles]
        # return torch.mean(weights * F.huber_loss(quantiles, newtarget, reduction='none'))
        return torch.mean(weights * F.smooth_l1_loss(quantiles, newtarget, reduction='none'))


    # 训练
    def train_step(self, state_batch, mcts_probs, value_batch, reward_batch, lr):
        """训练一次"""
        # 输入赋值       
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        value_batch = torch.FloatTensor(value_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        # 设置学习率
        # self.set_learning_rate(lr)

        # 前向传播
        self.policy_value_net.train()
        probs, values, rewards = self.policy_value_net(state_batch)

        value_loss = self.quantile_regression_loss(values, value_batch)
        policy_loss = F.cross_entropy(probs, mcts_probs)
        reward_loss = F.mse_loss(rewards.view(-1), reward_batch)

        # loss = policy_loss + value_loss/(value_loss/policy_loss).detach() + reward_loss/(reward_loss/policy_loss).detach() 
        loss = policy_loss + value_loss + reward_loss 
        # 参数梯度清零
        self.optimizer.zero_grad()
        # 反向传播并计算梯度
        loss.backward()
        # 更新参数
        self.optimizer.step()
        
        predicted_probs = torch.argmax(probs, dim=1)
        true_probs = torch.argmax(mcts_probs, dim=1)
        accuracy = (predicted_probs == true_probs).float().mean()
        return accuracy.item(), value_loss.item(), policy_loss.item(), reward_loss.item()
        

    # 保存模型
    def save_model(self, model_file):
        """ save model params to file """
        torch.save(self.policy_value_net.state_dict(), model_file)