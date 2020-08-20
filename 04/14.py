import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from IPython import display

# 游戏有设置最大步数 200/env._max_episode_steps 步
# unwarpped类是解除这个限制，玩多少步都可以
env = gym.make('CartPole-v0').unwrapped

# 随机玩一局
# env.step(0): 小车向左， env.step(1): 小车向右
# env.reset()
# for t in count(): 
#     env.render()
#     leftOrRight = random.randrange(env.action_space.n)
#     _, reward, done, _ = env.step(leftOrRight)
#     if done:
#         break


env.reset()


# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 状态、动作、下一个状态、打分
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# 定长的数组，自动按self.position循环更新数据
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def calc(self):
        reward = 0.0 
        for t in self.memory:
            if t.next_state!=None:
                reward += 1
        return reward/self.capacity

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # 线性输入连接的数量取决于conv2d层的输出，因此取决于输入图像的大小，因此请对其进行计算。
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # 使用一个元素调用以确定下一个操作，或在优化期间调用batch。返回tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))         #[B, 3, 40, 90] => [B, 16, 18, 43]
        x = F.relu(self.bn2(self.conv2(x)))         #[B, 16, 18, 43] => [B, 32, 7, 20]
        x = F.relu(self.bn3(self.conv3(x)))         #[B, 32, 7, 20] => [B, 32, 2, 8]
        return self.head(x.view(x.size(0), -1))     #[B, 512] => [B, 2]

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


# 获得当前小车的位置，转为正整数
# env.x_threshold x最大边距 [-2.4 ---- 0 ---- 2.4]
# env.state 当前状态 (位置x，x加速度, 偏移角度theta, 角加速度) 位置x可以为负数
def get_cart_location(screen_width):
    # 世界的总长度
    world_width = env.x_threshold * 2
    # 世界转屏幕像素系数
    scale = screen_width / world_width
    # 世界的中心点在屏幕中央，所以位置需要左偏移屏幕宽度的一半
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # gym要求的返回屏幕是400x600x3，但有时更大，如800x1200x3。 将其转换为torch order（CHW）。
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # cart位于下半部分，因此不包括屏幕的顶部和底部 [3, 400, 600]
    _, screen_height, screen_width = screen.shape
    # 把高度按160 - 320截断为160，[3, 160, 600]
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # 宽度只截取 60% ，左右各截取 30%
    view_width = int(screen_width * 0.6)

    # 获得当前小车的位置
    cart_location = get_cart_location(screen_width)

    if cart_location < view_width // 2:
        # 如果小车右边还有 30% 空间，则切片范围左边 60%【None, view_width, None】
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        # 如果小车左边有 30% 的空间，则切片范围最右边 60% [-view_width, None, None]
        slice_range = slice(-view_width, None)
    else:
        # 否则按小车的当前位置两端分别截取 30% 
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # 去掉边缘，使得我们有一个以cart为中心的方形图像
    screen = screen[:, :, slice_range]
    # 转换为float类型，重新缩放，转换为torch张量
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 调整大小并添加batch维度（BCHW）
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
# plt.show()


BATCH_SIZE = 512
# 得分的权重，这个值越小，越容易快速将得分压制到【0 ~ 1】之间，但同时最长远步骤的影响力也就越小，不能压制的太小
# 得分压制的太小会导致 Loss 过小，MSE的梯度会变得很小，不容易学习
GAMMA = 0.85
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10
MODEL_File = 'data/save/14_checkpoint.tar'


# 获取屏幕大小，以便我们可以根据AI gym返回的形状正确初始化图层。 
# 此时的典型尺寸接近3x40x90
# 这是get_screen（）中的限幅和缩小渲染缓冲区的结果
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape #【B，C，H，W】

# 从gym行动空间中获取行动数量 ， 就两种，左或右
n_actions = env.action_space.n

# 训练网络
policy_net = DQN(screen_height, screen_width, n_actions).to(device)

# 预测网络,相对于 policy_net 是上一次的训练参数
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 将学习率调到很小
optimizer = optim.RMSprop(policy_net.parameters(),lr=0.01)
memory = ReplayMemory(10000)

# 总共训练步数
steps_done = 0
# 平均一局步数
avg_step = 10
if os.path.exists(MODEL_File):
    checkpoint = torch.load(MODEL_File)
    policy_net_sd = checkpoint['policy_net']
    steps_done =  checkpoint['steps_done']
    avg_step = checkpoint['avg_step']
    policy_net.load_state_dict(policy_net_sd)
    target_net.load_state_dict(policy_net_sd)

# 开始随机动作，后期逐渐采用预测动作 【0.05 --> 0.9】返回动作shape: [B, 1]
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1)将返回每行的最大列值。 
            # 最大结果的第二列是找到最大元素的索引，因此我们选择具有较大预期奖励的行动。
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations():
    return
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 取100个episode的平均值并绘制它们
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 暂停一下，以便更新图表


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # 获得 [(state, action, next_state, reward), ...]
    transitions = memory.sample(BATCH_SIZE)
    # 转置batch（有关详细说明，请参阅https://stackoverflow.com/a/19343/3343043）。
    # 这会将过渡的batch数组转换为batch数组的过渡。
    # [T(a=1,b=2),T(a=1,b=2),T(a=1,b=2)] ==> T(a=(1,1,1),b=(2,2,2))  
    batch = Transition(*zip(*transitions))
    # 计算非最终状态的掩码并连接batch元素（最终状态将是模拟结束后的状态）
    # [True,True] Shape [128]
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                      batch.next_state)), device=device, dtype=torch.bool)
    # [121, 3, 40, 90]
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # [128, 3, 40, 90]
    state_batch = torch.cat(batch.state)
    # [[1],[1]] Shape [128, 1]
    action_batch = torch.cat(batch.action)
    # [1, 1] Shape [128]
    reward_batch = torch.cat(batch.reward)
    
    # 计算Q(s_t，a) - 模型计算Q(s_t)，然后我们选择所采取的动作列。
    # 这些是根据policy_net对每个batch状态采取的操作
    # 根据当前的动作获得当前动作对应的得分 
    # [[6.2464],[3.2442]] shape [128,1]
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # 计算所有下一个状态的V(s_{t+1})
    # non_final_next_states的操作的预期值是基于“较旧的”target_net计算的; 
    # 因为涉及到模型BatchNorm2d的参数，需要采用eval()，而当前模型 policy_net 又处理train()状态，所以只能另外开一个 target_net 来计算
    # 用 max(1)[0] 选择一下个状态最佳得分。这是基于掩码合并的，这样我们就可以得到预期的得分，或者在状态是最终的情况下为0。
    # 预测下一个状态的最佳得分，如果没有下一步，则下一步的概率为0 ,shape : 121
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # [6.4941, 0.0000] Shape [128]
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 用预期的下一步的最佳得分*衰减，再加上本次的奖励获得总得分
    # [6.8447, 0] Shape [128]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算Huber损失
    # 期望当前状态得分和下一次状态得分一样，这样，如果下一次状态得分高，则鼓励当前动作，否则压制当前动作
    # 为了最大限度地降低此错误，我们将使用Huber损失。当误差很小时，Huber损失就像均方误差一样，
    # 但是当误差很大时，就像平均绝对误差一样 - 当估计噪声很多时，这使得它对异常值更加鲁棒。
    # 在 [-1, 1] 区间，直接采用MSE: 1/2*(y-f(x))^2，其余采用L1Loss: |y-f(x)| 
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-5, 5)
    optimizer.step()
    return loss

num_episodes = 5000000
for i_episode in range(num_episodes):
    # 初始化环境和状态
    env.reset()
    last_screen = get_screen()                  # [1, 3, 40, 90]
    current_screen = get_screen()               # [1, 3, 40, 90]
    state = current_screen - last_screen        # [1, 3, 40, 90]   
    reward_proportion = memory.calc() 
    for t in count():
        # 选择动作并执行
        action = select_action(state)
        observation_, _reward, done, _ = env.step(action.item())
        if done: _reward = 0.0

        # # 不采用系统默认的reward，太难学习了
        # x, x_dot, theta, theta_dot = observation_   
        # # r1代表车的 x水平位移 与 x最大边距 的距离差的得分
        # r1 = math.exp((env.x_threshold - abs(x))/env.x_threshold) - math.exp(1)/2
        # # r2代表棒子的 theta离垂直的角度 与 theta最大角度 的差的得分
        # r2 = math.exp((env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians) - math.exp(1)/2
        # # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度。
        # _reward = r1 + r2   

        reward = torch.tensor([_reward], device=device)

        # 观察新的状态,下一个状态 等于当前屏幕 - 上一个屏幕 ？ 这样抗干扰高？所有的状态预测都是像素差
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 在记忆中存储过渡,但减少为1的奖励
        if random.random()<reward_proportion and not done and steps_done>memory.capacity:
            if len(memory.memory)==memory.capacity: 
                while memory.memory[memory.position].next_state==None:
                    memory.position = (memory.position + 1) % memory.capacity
        memory.push(state, action, next_state, reward)

        # 移动到下一个状态
        state = next_state

        # 执行优化的一个步骤（在目标网络上）
        loss = optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    avg_step = avg_step*0.99 + t*0.01 

    # 更新目标网络，复制DQN中的所有权重和偏差
    if i_episode % TARGET_UPDATE == 0 and loss!=None :
        target_net.load_state_dict(policy_net.state_dict())
        print(i_episode, steps_done, t, '/' , avg_step, "loss:", loss.item(), "reward_proportion", \
            reward_proportion, "position:",memory.position,"eps_threshold:",EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY))
        torch.save({    'policy_net': policy_net.state_dict(),
                    'steps_done': steps_done,
                    'avg_step': avg_step,
                }, MODEL_File)

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()