import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

# 网络模型

# 定义残差块，删除了BN
class ResidualBlock(nn.Module):
    #实现子module: Residual    Block
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
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
        # 直接来2个残差网络
        self.conv1=self._make_layer(4, 64, 3)
        self.conv2=self._make_layer(64, 128, 3)

        # 动作预测
        self.act_conv1 = nn.Conv2d(128, 4, 1)
        self.act_fc1 = nn.Linear(4*size*size, size*size)
        # 动作价值
        self.val_conv1 = nn.Conv2d(128, 2, 1)
        self.val_fc1 = nn.Linear(2*size*size, 64)
        self.val_fc2 = nn.Linear(64, 1)

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
        x = self.conv1(x)
        x = self.conv2(x)
        # 动作
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(x.size(0), -1)
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)

        # 胜率 输出为 -1 ~ 1 之间的数字
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(x.size(0), -1)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    # 设置学习率
    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def print_netwark(self):
        x=torch.Tensor(1,4,self.size,self.size).to(self.device)
        print(self.policy_value_net)
        v,p=self.policy_value_net(x)
        print("value:",v.size())
        print("policy:",p.size())

    def __init__(self, size, model_file=None, device=None, l2_const=1e-4):
        self.size = size
        self.device=device
        self.l2_const = l2_const  
        self.policy_value_net = Net(size).to(device)

        self.print_netwark()

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file and os.path.exists(model_file):
            print("Loading model", model_file)
            net_sd = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_sd)

    # 根据当前状态得到，action的概率和概率
    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = torch.FloatTensor(state_batch).to(self.device) 
        log_act_probs, value = self.policy_value_net(state_batch)
        # 还原成标准的概率
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    # 从当前棋局获得 ((action, act_probs),...) 的可用动作+概率和当前棋盘胜率
    def policy_value_fn(self, game):
        """
        input: game
        output: a list of (action, probability) tuples for each available
        action and the score of the game state
        """
        legal_positions = game.actions_to_positions(game.availables)
        current_state = game.current_state().reshape(1, 4, self.size, self.size)
        act_probs, value = self.policy_value(current_state)
        act_probs = act_probs.flatten()

        actions = game.positions_to_actions(legal_positions)
        
        act_probs = zip(actions, act_probs[legal_positions])
        value = value[0,0]
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
        for name, parms in self.policy_value_net.named_parameters():
            print('name:', name, 'grad_requirs:', parms.requires_grad,' grad_value:',torch.max(parms.grad))
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