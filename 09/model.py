import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from collections import OrderedDict
from torchvision.models import resnet34

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

    def __getitem__(self, key):
        try:
            value = super().__getitem__(key)
        except KeyError:
            raise
        else:
            self.move_to_end(key)
            return value


class MlpBlock(nn.Module):
    def __init__(self, mlp_dim:int, hidden_dim:int):
        super(MlpBlock, self).__init__()
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.Linear2 = nn.Linear(hidden_dim, mlp_dim)
    def forward(self,x):
        y = self.Linear1(x)
        y = self.gelu(y)
        y = self.Linear2(y)
        return y

# 混合感知机块
# 输入 x： (n_samples, n_patches, hidden_dim)
# 输出 ： 和输入 x 的张量保存一致
class MixerBlock(nn.Module):
    def __init__(self, n_patches: int , hidden_dim: int, token_dim: int, channel_dim: int):
        super(MixerBlock, self).__init__()
        self.MLP_block_token = MlpBlock(n_patches, token_dim)
        self.MLP_block_chan = MlpBlock(hidden_dim, channel_dim)
        self.LayerNorm_token = nn.LayerNorm(hidden_dim)
        self.LayerNorm_chan = nn.LayerNorm(hidden_dim)

    def forward(self,x):
        # 针对 n_patches 做全连接(token)
        y = self.LayerNorm_token(x)           # (n_samples, n_patches, hidden_dim)
        y = y.permute(0, 2, 1)          # (n_samples, hidden_dim, n_patches)
        y = self.MLP_block_token(y)     # (n_samples, hidden_dim, n_patches)
        y = y.permute(0, 2, 1)          # (n_samples, n_patches, hidden_dim)
        x = x + y   # (n_samples, n_patches, hidden_dim)
        # 针对 hidden_dim 做全连接(channel)
        y = self.LayerNorm_chan(x)  # (n_samples, n_patches, hidden_dim)
        y = self.MLP_block_chan(y) # (n_samples, n_patches, hidden_dim)
        return x + y

# 混合多层感知机网络
# 输入 x (n_samples, n_channels, image_size, image_size)
# 输出 逻辑分类张量 (n_samples, n_classes)
# 构造函数：
# image_size  : 输入图片的边长
# n_channels  : 输入图片的层数
# patch_size  : 图片分割边长，是 image_size 的约数， n_patches 为分割的块数 为（图片边长/分割边长）的平方
# hidden_dim  : 每个图片块的最后维度
# token_dim   : token 混合的维度
# channel_dim : channel 混合的维度
# n_classes   : 输出类别个数
# n_blocks    : 多少个模型块相当于残差的层数
class MLP_Mixer(nn.Module):
    def __init__(self, image_size_h, image_size_w, n_channels, patch_size_h, patch_size_w, hidden_dim, token_dim, channel_dim, n_action, n_blocks, dropout = 0.):
        super(MLP_Mixer, self).__init__()
        n_patches =(image_size_h//patch_size_h) * (image_size_w//patch_size_w) # image_size 可以整除 patch_size
        self.patch_size_embbeder = nn.Conv2d(kernel_size=(patch_size_h,patch_size_w), stride=(patch_size_h,patch_size_w), in_channels=n_channels, out_channels= hidden_dim)
        self.blocks = nn.ModuleList([
            MixerBlock(n_patches=n_patches, hidden_dim=hidden_dim, token_dim=token_dim, channel_dim=channel_dim) for i in range(n_blocks)
        ])
        self.drop_layer = nn.Dropout(p=dropout)

        self.flatten = nn.Flatten(start_dim=2)
        self.action_line = nn.Linear(hidden_dim, hidden_dim)
        self.value_line = nn.Linear(hidden_dim, hidden_dim)
        self.Layernorm1 = nn.LayerNorm(hidden_dim)  # (n_samples, n_patches, hidden_dim)
        self.Layernorm2 = nn.LayerNorm(hidden_dim)  # (n_samples, n_patches, hidden_dim)
        self.action_fc = nn.Linear(hidden_dim, n_action)
        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self,x):
        x = self.drop_layer(x)
        x = self.patch_size_embbeder(x) # (n_samples, hidden_dim, image_size/patch_size, image_size/patch_size)
        x = self.flatten(x)         # (n_samples, hidden_dim, n_patches)
        x = x.permute(0, 2, 1)      # (n_samples, n_patches, hidden_dim)
        for block in self.blocks:
            x = block(x)            # (n_samples, n_patches, hidden_dim)

        x_act = self.action_line(x)         # (n_samples, n_patches, hidden_dim)
        x_act = self.Layernorm1(x_act)      # (n_samples, n_patches, hidden_dim)
        x_act = x_act.mean(dim = 1)         # (n_sample, hidden_dim)
        x_action = F.log_softmax(self.action_fc(x_act), dim=1)  # (n_samples, n_action)

        x_val = self.value_line(x)          # (n_samples, n_patches, hidden_dim)
        x_val = self.Layernorm2(x_val)      # (n_samples, n_patches, hidden_dim)
        x_val = x_val.mean(dim = 1)         # (n_sample, hidden_dim)
        x_val = F.gelu(x_val)               # (n_samples, hidden_dim)
        x_val = F.gelu(self.value_fc1(x_val))
        x_value = torch.tanh(self.value_fc2(x_val))

        return x_action, x_value

# 定义残差块
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
    def __init__(self, image_size, action_size):
        super(Net, self).__init__()

        # 由于每个棋盘大小对最终对应一个动作，所以补齐的效果比较好
        # 直接来9层的残差网络
        self.conv=self._make_layer(9, 128, 9)

        # 动作预测
        self.act_conv1 = nn.Conv2d(128, 1, 1)
        self.act_conv1_bn = nn.BatchNorm2d(1)
        self.act_fc1 = nn.Linear(image_size, action_size)
        # 动作价值
        self.val_conv1 = nn.Conv2d(128, 1, 1)
        self.val_conv1_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(image_size, 128)
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
        x = self.conv(x)

        # 动作
        x_act = self.act_conv1(x)
        x_act = self.act_conv1_bn(x_act)
        x_act = x_act.view(x_act.size(0), -1)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # 胜率 输出为 -1 ~ 1 之间的数字
        x_val = self.val_conv1(x)
        x_val = self.val_conv1_bn(x_val)
        x_val = F.relu(x_val)
        x_val = x_val.view(x_val.size(0), -1)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class ResNet(nn.Module):
    def __init__(self, image_size, action_size):
        super(ResNet, self).__init__()

        resnet = resnet34()
        resnet.conv1 = nn.Conv2d(9, 64, kernel_size=3, bias=False)
        num_ftrs = resnet.fc.in_features
        self.conv = nn.Sequential(*(list(resnet.children())[:-2]))

        # 动作预测
        self.act_conv1 = nn.Conv2d(num_ftrs, image_size, (2,1))
        # self.act_conv1_bn = nn.BatchNorm2d(image_size)
        self.act_fc1 = nn.Linear(image_size, action_size)
        # 动作价值
        self.val_conv1 = nn.Conv2d(num_ftrs, image_size, (2,1))
        # self.val_conv1_bn = nn.BatchNorm2d(image_size)
        self.val_fc1 = nn.Linear(image_size, image_size)
        self.val_fc2 = nn.Linear(image_size, 1)


    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        # 动作
        x_act = self.act_conv1(x)
        #print(x_act.shape)

        # x_act = self.act_conv1_bn(x_act)
        x_act = x_act.view(x_act.size(0), -1)
        #print(x_act.shape)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # 胜率 输出为 -1 ~ 1 之间的数字
        x_val = self.val_conv1(x)
        # x_val = self.val_conv1_bn(x_val)
        x_val = F.relu(x_val)
        x_val = x_val.view(x_val.size(0), -1)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyValueNet():
    def __init__(self, input_width, input_height, output_size, model_file=None, device=None, l2_const=1e-4):
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_width * input_height
        self.output_size = output_size
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        print("use", device)

        self.l2_const = l2_const  
        self.policy_value_net = ResNet(self.input_size, self.output_size)
        # self.policy_value_net = Net(self.input_size, self.output_size)
        # self.policy_value_net = MLP_Mixer(20,10,9,2,5,128,64,512,5,8,dropout=0.1)
        self.policy_value_net.to(device)
        self.print_netwark()

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)       
        # self.optimizer = optim.SGD(self.policy_value_net.parameters(), lr=1e-3, momentum=0.9, weight_decay=self.l2_const)

        self.load_model_file=False
        if model_file and os.path.exists(model_file):
            print("Loading model", model_file)
            net_sd = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_sd)
            self.load_model_file = True

        self.cache = Cache(maxsize=500000)

    # 设置学习率
    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # 打印当前网络
    def print_netwark(self):
        x = torch.Tensor(1,9,20,10).to(self.device)
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

        key = game.get_key()
        if key in self.cache:
            act_probs, value = self.cache[key] 
        else:
            if self.load_model_file:
                current_state = game.current_state().reshape(1, -1, self.input_height, self.input_width)
                act_probs, value = self.policy_value(current_state)
                act_probs = act_probs.flatten()
            else:
                act_len=game.actions_num
                act_probs=np.ones([act_len])/act_len
                value = np.array([[0.]])
            actions = game.availables
            act_probs = list(zip(actions, act_probs[actions]))
            value = value[0,0]

            self.cache[key] = (act_probs, value) 
        
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

        # indices = torch.LongTensor([3]).to(self.device)
        # mask = torch.index_select(state_batch,1,indices)  # 取出mask层 [batchsize, 1, 10, 20]
        # mask = torch.mean(mask,[1,2,3])  # 确定mask [0,1,0,0....1] shape: [batchsize]
        # policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1) * mask)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        # print(mask.data.cpu().numpy())
        # print(mcts_probs.data.cpu().numpy())
        # print(torch.sum(mcts_probs * log_act_probs, 1).data.cpu().numpy())
        # print((torch.sum(mcts_probs * log_act_probs, 1) * mask).data.cpu().numpy())
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