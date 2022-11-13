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
    def __init__(self, input_width, input_height, output_size, model_file=None, device=None, l2_const=1e-2):
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_width * input_height
        self.output_size = output_size
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        print("use", device)

        self.l2_const = l2_const  
        # ViT-Ti : depth 12 width 192 heads 3
        self.policy_value_net = VitNet(embed_dim=192, depth=12, num_heads=3, num_classes=4, num_quantiles=8, drop_ratio=0., drop_path_ratio=0., attn_drop_ratio=0.)
        # ViT-S : depth 12 width 386 heads 6
        # self.policy_value_net = VitNet(embed_dim=386, depth=12, num_heads=6, num_classes=4, num_quantiles=12)
        # ViT-B : depth 12 width 768 heads 12
        # self.policy_value_net = VitNet(num_classes=4, num_quantiles=12)
        self.policy_value_net.to(device)
        # self.print_netwark()

        self.optimizer = optim.AdamW(self.policy_value_net.parameters(), lr=1e-6, weight_decay=self.l2_const)       
        # self.optimizer = optim.SGD(self.policy_value_net.parameters(), lr=1e-6, momentum=0.9, weight_decay=self.l2_const)

        self.load_model_file=False
        if model_file and os.path.exists(model_file):
            print("Loading model", model_file)
            net_sd = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_sd)
            self.load_model_file = True

        self.lr = 0

    # 设置学习率
    def set_learning_rate(self, lr):
        if self.lr != lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
            print("Set modle learn rate to:", lr)

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
        with torch.no_grad(): 
            act_probs, value = self.policy_value_net.forward(state_batch_tensor)    #[b, num_classes] [b, num_quantiles]
            
        act_probs = torch.softmax(act_probs,dim=1)
        num_quantiles = value.shape[1]
        num_value =  int(num_quantiles * 0.75)
        value =  torch.mean(value[:, num_value:] , dim=1)

        # 还原成标准的概率
        act_probs = act_probs.cpu().numpy()
        value = value.cpu().numpy()

        return act_probs, value

    # 从当前游戏获得 ((action, act_probs),...) 的可用动作+概率和当前游戏胜率
    def policy_value_fn(self, game):
        """
        输入: 游戏
        输出: 一组（动作， 概率）和游戏当前状态的胜率
        """       
        if self.load_model_file:
            current_state = game.current_state().reshape(1, -1, self.input_height, self.input_width)
            act_probs, value = self.policy_value(current_state)
            act_probs=act_probs[0]
            value=float(value[0])
        else:
            act_len=game.actions_num
            act_probs=np.ones([act_len])/act_len
            value = 0.
        
        actions = game.availables
        act_probs = list(zip(actions, act_probs[actions]))

        return act_probs, value

    # 价值网络损失
    def quantile_regression_loss(self, quantiles, target):
        num_quantiles = quantiles.shape[1]
        tau = (torch.arange(num_quantiles) + 0.5) / num_quantiles                   #[b, num_quantiles]
        target = target.unsqueeze(1)                                                #[b, 1]
        # target = target.repeat(1, num_quantiles)                                    #[b, num_quantiles]
        weights = torch.where(quantiles > target, tau, 1 - tau)                     #[b, num_quantiles]
        return torch.mean(weights * F.huber_loss(quantiles, target, reduction='none'))

    # 训练
    def train_step(self, state_batch, mcts_probs, value_batch, lr):
        """训练一次"""
        # 输入赋值       
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        value_batch = torch.FloatTensor(value_batch).to(self.device)

        # 设置学习率
        self.set_learning_rate(lr)

        # 前向传播
        self.policy_value_net.train()
        probs, values = self.policy_value_net(state_batch)

        value_loss = self.quantile_regression_loss(values, value_batch)
        policy_loss = F.cross_entropy(probs, mcts_probs)

        loss = value_loss + policy_loss

        # 参数梯度清零
        self.optimizer.zero_grad()
        # 反向传播并更新
        loss.backward()
        self.optimizer.step()
                
        return loss.item(), value_loss.item(), policy_loss.item()

    # 保存模型
    def save_model(self, model_file):
        """ save model params to file """
        torch.save(self.policy_value_net.state_dict(), model_file)