import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

# 定义残差块
class ResidualBlock(nn.Module):
    #实现子module: Residual    Block
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
        return F.leaky_relu(out)

class Net(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.conv1=self._make_layer(3, 64, 4)
        # self.conv2=self._make_layer(64, 64, 3)

        # 动作预测
        self.act_conv1 = nn.Conv2d(64, 2, 1)
        self.act_fc1 = nn.Linear(2*input_size, input_size)
        self.act_fc2 = nn.Linear(input_size, output_size)

        # 动作价值
        self.val_conv1 = nn.Conv2d(64, 2, 1)
        self.val_fc1 = nn.Linear(2*input_size, input_size)
        self.val_fc2 = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)

        x_act = F.leaky_relu(self.act_conv1(x))
        x_act = x_act.view(x.size(0), -1)
        x_act = F.leaky_relu(self.act_fc1(x_act))
        x_act = F.log_softmax(self.act_fc2(x_act),dim=1)

        x_val = F.leaky_relu(self.val_conv1(x))
        x_val = x_val.view(x.size(0), -1)
        x_val = F.leaky_relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

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


class PolicyValueNet():
    def __init__(self, input_width, input_height, output_size, model_file=None, device=None, l2_const=1e-4):
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_width * input_height
        self.output_size = output_size
        self.device=device
        self.l2_const = l2_const  
        self.policy_value_net = Net(self.input_size, self.output_size).to(device)
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
        x = torch.Tensor(1,3,20,10).to(self.device)
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
        # 由于样本不足，导致单张局面做预测时的分布与平均分布相差很大，会出现无法预测的情况，所以不加 eval() 锁定bn为平均方差
        # 或者 设置 BN 的 track_running_stats=False ，不使用全局的方差，直接用每批的方差来标准化。
        with torch.no_grad(): 
            log_act_probs, value = self.policy_value_net.forward(state_batch_tensor)
        # 还原成标准的概率
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        value = value.data.cpu().numpy()

        return act_probs, value

    # 从当前游戏获得 ((action, act_probs),...) 的可用动作+概率和当前游戏胜率
    def policy_value_fn(self, game):
        """
        输入: 游戏
        输出: 一组（动作， 概率）和游戏当前状态的胜率
        """
        current_state = game.current_state().reshape(1, -1, self.input_height, self.input_width)
        act_probs, value = self.policy_value(current_state)

        act_probs = act_probs.flatten()
        actions = game.availables()
        act_probs = list(zip(actions, act_probs[actions]))
        value = value[0,0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """训练一次"""
        # 输入赋值       
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        winner_batch = torch.FloatTensor(winner_batch).to(self.device)

        # 参数梯度清零
        self.optimizer.zero_grad()
        # 设置学习率
        self.set_learning_rate(lr)

        # 前向传播
        self.policy_value_net.train()
        log_act_probs, value = self.policy_value_net(state_batch)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        # 胜率
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

        # 反向传播并更新
        loss.backward()
        self.optimizer.step()

        # if random.random()>0.99:
        #     print(loss, value_loss, policy_loss)
        # for name, parms in self.policy_value_net.named_parameters():
        #     grad_value = torch.max(parms.grad)
        #     print('name:', name, 'grad_requirs:', parms.requires_grad,' grad_value:',grad_value)
        # raise "ss"

        # 计算信息熵，越小越好, 只用于监控
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), value_loss.item(), policy_loss.item(), entropy.item()

    # 保存模型
    def save_model(self, model_file):
        """ save model params to file """
        torch.save(self.policy_value_net.state_dict(), model_file)