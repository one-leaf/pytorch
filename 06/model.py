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
    #实现子module: Residual Block
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right=shortcut
        
    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out)

class Net(nn.Module):

    def __init__(self, size):
        super(Net, self).__init__()

        # 由于每个棋盘大小对最终对应一个动作，所以补齐的效果比较好
        # 直接来18层的残差网络
        self.first_conv = nn.Conv2d(11,128,3,1,1, bias=False)
        self.first_conv_bn = nn.BatchNorm2d(128)

        self.conv=self._make_layer(128, 128, 19)

        # 动作预测
        self.act_conv1 = nn.Conv2d(128, 2, 1)
        self.act_conv1_bn = nn.BatchNorm2d(2)
        self.act_fc1 = nn.Linear(2*size*size, size*size)
        # 动作价值
        self.val_conv1 = nn.Conv2d(128, 1, 1)
        self.val_conv1_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(size*size, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        #构建layer,包含多个residual block
        shortcut=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
 
        layers=[ ]
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
        
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_conv_bn(x)
        x = F.leaky_relu(x)

        x = self.conv(x)

        # 动作
        x_act = self.act_conv1(x)
        x_act = self.act_conv1_bn(x_act)
        x_act = F.leaky_relu(x_act)
        x_act = x_act.view(x_act.size(0), -1)
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)

        # 胜率 输出为 -1 ~ 1 之间的数字
        x_val = self.val_conv1(x)
        x_val = self.val_conv1_bn(x_val)
        x_val = F.leaky_relu(x_val)
        x_val = x_val.view(x_val.size(0), -1)
        x_val = F.leaky_relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    def __init__(self, size, model_file=None, device=None, l2_const=1e-4):
        self.size = size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        self.l2_const = l2_const  
        self.policy_value_net = Net(size).to(device)

        self.cache = Cache(maxsize=100000)
        self.print_netwark()

        # self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)       
        self.optimizer = optim.SGD(self.policy_value_net.parameters(), lr=1e-3, momentum=0.9, weight_decay=self.l2_const)

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
        x=torch.Tensor(1,11,self.size,self.size).to(self.device)
        print(self.policy_value_net)
        v,p=self.policy_value_net(x)
        print("value:",v.size())
        print("policy:",p.size())

    # 根据当前状态得到，action的概率和胜率
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

        return act_probs, value

    # 从当前棋局获得 ((action, act_probs),...) 的可用动作+概率和当前棋盘胜率
    def policy_value_fn(self, game):
        """
        input: game
        output: a list of (action, probability) tuples for each available
        action and the score of the game state
        """
        # 只缓存前10步棋
        max_cache_step = 10
        if len(game.actions)<=max_cache_step:
            key = game.get_key()
            if key in self.cache:
                return self.cache[key]

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

        # print(loss, value_loss, policy_loss)
        # for name, parms in self.policy_value_net.named_parameters():
        #     grad_value = torch.max(parms.grad)
        #     print('name:', name, 'grad_requirs:', parms.requires_grad,' grad_value:',grad_value)
        # raise "ss"

        self.optimizer.step()
        # 计算信息熵，越小越好, 只用于监控
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), value_loss.item(), policy_loss.item(), entropy.item()

    # 保存模型
    def save_model(self, model_file):
        """ save model params to file """
        torch.save(self.policy_value_net.state_dict(), model_file)