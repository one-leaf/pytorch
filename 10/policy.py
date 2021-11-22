import random
import numpy as np
import torch
import utils
import torch.optim as optim

import torch.nn as nn

# 如果有显卡就用显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算所有的奖励按0.99衰减
def calculate_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

# 低维度转高维度
class PositionalMapping(nn.Module):
    """
    位置映射层.
    讲连续的一维数据转为高维数据.
    参考论文地址 (https://arxiv.org/pdf/2003.08934.pdf)
    """

    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):

        x = x * self.scale

        if self.L == 0:
            return x

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):
    """
    包含了位置特征的多层感知机
    本例中输入有 8 个维度（x，y，vx，vy，theta，vtheta，step_id，phi）
    位置化后，为 7*8+1 = 57 个维度
    加上四层的全连接，输出 9 个动作的特征，注意不是概率
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mapping = PositionalMapping(input_dim=input_dim, L=7)

        h_dim = 128
        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # shape x: 1 x m_token x m_state
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class ActorCritic(nn.Module):
    """
    火箭模型的策略和更新
    两个模型，第一个输出动作的概率，第二个输出当前得分，模型并不共用
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.output_dim = output_dim
        self.actor = MLP(input_dim=input_dim, output_dim=output_dim)
        self.critic = MLP(input_dim=input_dim, output_dim=1)
        self.softmax = nn.Softmax(dim=-1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=5e-5)

    # 前向传播 x : (batch_size, state) ==> (1, 8)
    def forward(self, x):
        y = self.actor(x)
        probs = self.softmax(y)
        value = self.critic(x)
        return probs, value

    # 获得动作
    def get_action(self, state, deterministic=False, exploration=0.01):
        # 增加 1 的维度代替 batch_size (8) ==> (1,8)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, value = self.forward(state)
        # 只要第一个返回的概率 (9)
        probs = probs[0, :]
        value = value[0]

        # 如果是确定输出，直接取最大值的位置
        if deterministic:
            action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        else:
            # 否则0.01的概率随机选择，0.99的概率按模型输出的概率选择
            if random.random() < exploration:  # exploration
                action_id = random.randint(0, self.output_dim - 1)
            else:
                action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))

        # 输出概率的对数，方便后续的反向传播
        log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, log_prob, value

    # 反向传播
    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=0.99):

        # 计算Q值, Qval: 本局最终的状态得分；
        #         rewards：当前步的得分；
        #         masks：最后一步是否结束
        # 将 后一步的 reward 按照 0.99 的衰减，叠加到 上一步的 reward 上
        # 倒序循环
        # R = rewards[step] + gamma * R * masks[step]
        # 输出 [..R] => [..Q],
        # 注意:
        #       如果最后一步结束了，R的初始取值直接是 rewards[-1]
        #       如果到最后游戏还没有结束，最后的得分，就需要加上预测的Qval*0.99，作为巨大的奖励提升
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        # 本局的所有步数的动作对数概率和打分
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        #当前实际的得分 - 上一步预测的值，
        advantage = Qvals - values
        # 对数概率 与实际得分与预测得分 取负
        actor_loss = (-log_probs * advantage.detach()).mean()
        # 对得分差异取F2范数*0.5，强制降低差异
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        # 梯度清零，反向更新梯度，更新参数
        network.optimizer.zero_grad()
        ac_loss.backward()
        network.optimizer.step()

