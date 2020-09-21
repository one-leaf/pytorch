import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import random
from threading import Lock
from collections import OrderedDict

class Cache(OrderedDict):
    def __init__(self, maxsize=128, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

# 网络模型
# 定义残差块，固定住BN的方差和均值
class ResidualBlock(nn.Module):
    #实现子module: Residual    Block
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=False)
        )
        self.right=shortcut
        
    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.leaky_relu(out)

class Net(nn.Module):

    def __init__(self, size):
        super(Net, self).__init__()

        # 由于每个棋盘大小对最终对应一个动作，所以补齐的效果比较好
        # 直接来2个残差网络
        self.conv1=self._make_layer(7, 64, 4)
        # self.conv2=self._make_layer(64, 128, 3)
        # self.conv3=self._make_layer(128, 128, 3)

        # 动作预测
        self.act_conv1 = nn.Conv2d(64, 4, 1)
        self.act_fc1 = nn.Linear(4*size*size, 2*size*size)
        self.act_fc2 = nn.Linear(2*size*size, size*size)
        # 动作价值
        self.val_conv1 = nn.Conv2d(64, 2, 1)
        self.val_fc1 = nn.Linear(2*size*size, size*size)
        self.val_fc2 = nn.Linear(size*size, 1)
        # self.val_fc3 = nn.Linear(64, 1)

    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        #构建layer,包含多个residual block
        shortcut=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=False)
        )
 
        layers=[ ]
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)

        # 动作
        x_act = F.leaky_relu(self.act_conv1(x))
        x_act = x_act.view(x.size(0), -1)
        x_act = F.leaky_relu(self.act_fc1(x_act))
        x_act = F.log_softmax(self.act_fc2(x_act),dim=1)

        # 胜率 输出为 -1 ~ 1 之间的数字
        x_val = F.leaky_relu(self.val_conv1(x))
        x_val = x_val.view(x.size(0), -1)
        x_val = F.leaky_relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    def __init__(self, size, model_file=None, device=None, l2_const=1e-4):
        self.size = size
        self.device=device
        self.l2_const = l2_const  
        self.policy_value_net = Net(size).to(device)

        self.cache = Cache(maxsize=10000)
        self.print_netwark()

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)       

        if model_file and os.path.exists(model_file):
            print("Loading model", model_file)
            net_sd = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_sd)

    # 设置学习率
    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # 打印当前网络
    def print_netwark(self):
        x=torch.Tensor(1,7,self.size,self.size).to(self.device)
        print(self.policy_value_net)
        v,p=self.policy_value_net(x)
        print("value:",v.size())
        print("policy:",p.size())

    # 根据当前状态得到，action的概率和概率
    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if torch.is_tensor(state_batch):
            state_batch_tensor = state_batch.to(self.device)
        else:
            state_batch_tensor = torch.FloatTensor(state_batch).to(self.device)

        self.policy_value_net.eval()
        # 由于样本不足，导致单张局面做预测时的分布与平均分布相差很大，会出现无法预测的情况，所以不加 eval() 锁定bn为平均方差
        # 或者 设置 BN 的 track_running_stats=False ，不使用全局的方差，直接用每批的方差来标准化。
        with torch.no_grad(): 
            log_act_probs, value = self.policy_value_net.forward(state_batch_tensor)
        # 还原成标准的概率
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        value = value.data.cpu().numpy()

        # if random.random()<0.0001:
        #     idx = np.argmax(act_probs)
        #     print("state var:",state_batch_tensor.var(),"probs max:", np.max(act_probs), "probs min:", np.min(act_probs), \
        #         "act:", (idx%self.size, self.size-(idx//self.size)-1), "value:", value)

        return act_probs, value

    # 从当前棋局获得 ((action, act_probs),...) 的可用动作+概率和当前棋盘胜率
    def policy_value_fn(self, game):
        """
        input: game
        output: a list of (action, probability) tuples for each available
        action and the score of the game state
        """
        # 只缓存前3步棋
        max_cache_step = 3
        if len(game.actions)<=max_cache_step:
            key = ",".join([str(x*game.size+y) for x,y in game.actions])
            if key in self.cache:
                return self.cache[key]

        # square_state, availables, actions = game.current_and_next_state()
        # act_probs_list, value_list = self.policy_value(square_state)
        # for i, act in enumerate(availables):
        #     _legal_positions = game.actions_to_positions(act)
        #     _key = ",".join([str(x*game.size+y) for x,y in actions[i]])
        #     act_probs = act_probs_list[i]
        #     act_probs = act_probs.flatten()
        #     value = value_list[i]
        #     # actions = game.positions_to_actions(_legal_positions)
        #     # 这里zip需要转为list放入cache，否则后续会返回为[]
        #     act_probs_zip = list(zip(act, act_probs[_legal_positions]))
        #     value = value[0]
        #     self.cache[_key] = (act_probs_zip, value)       
        # return self.cache[key]

        legal_positions = game.actions_to_positions(game.availables)
        current_state = game.current_state().reshape(1, -1, self.size, self.size)
        act_probs, value = self.policy_value(current_state)
        act_probs = act_probs.flatten()
        actions = game.positions_to_actions(legal_positions)
        act_probs = list(zip(actions, act_probs[legal_positions]))
        value = value[0,0]

        if len(game.actions)<=max_cache_step:
            self.cache[key] = (act_probs, value) 
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable       
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        winner_batch = torch.FloatTensor(winner_batch).to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        self.set_learning_rate(lr)

        # forward
        self.policy_value_net.train()
        log_act_probs, value = self.policy_value_net(state_batch)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        # 胜率
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

        # backward and optimize
        loss.backward()

        print(loss, value_loss, policy_loss)
        # for name, parms in self.policy_value_net.named_parameters():
        #     grad_value = torch.max(parms.grad)
        #     print('name:', name, 'grad_requirs:', parms.requires_grad,' grad_value:',grad_value)
        # raise "ss"

        self.optimizer.step()
        # 计算信息熵，越小越好, 只用于监控
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()

    # 保存模型
    def save_model(self, model_file):
        """ save model params to file """
        torch.save(self.policy_value_net.state_dict(), model_file)