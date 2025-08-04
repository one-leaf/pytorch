from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样智能体的交互数据"""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        """添加一条转移数据到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        """随机采样一批数据"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        """返回缓冲区当前数据量"""
        return len(self.buffer)

def moving_average(a, window_size):
    """计算滑动平均，用于平滑曲线"""
    # a = [3, 5, 7, 1, 9]，窗口大小 window_size = 3
    # np.insert(a, 0, 0) → [0, 3, 5, 7, 1, 9]
    # np.cumsum(...) → [0, 3, 8, 15, 16, 25]
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    # cumulative_sum[window_size:] → [15, 16, 25]
    # cumulative_sum[:-window_size] → [0, 3, 8]
    # 差值：[15-0, 16-3, 25-8] = [15, 13, 17]
    # 除以窗口：[15/3, 13/3, 17/3] ≈ [5.0, 4.333, 5.667]
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    # 计算开始和结束部分的平均值
    # window_size-1 = 2
    # np.arange(1, 2, 2) → [1]
    r = np.arange(1, window_size-1, 2)
    # 计算开始部分的平均值
    # a[:window_size-1] → [3, 5]
    # np.cumsum(...) → [3, 8]
    # [::2] → 取索引 0 的元素 → [3]
    # 除以 r → [3/1] = [3.0]  
    begin = np.cumsum(a[:window_size-1])[::2] / r
    # 计算结束部分的平均值
    # a[:-window_size:-1] → 从末尾反向取到倒数 window_size 处 → [9, 1]
    # np.cumsum(...) → [9, 10]
    # [::2] → 取索引 0 的元素 → [9]
    # 除以 r → [9/1] = [9.0]
    # [::-1] → 反转数组 → [9.0]（不变）
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    # 将开始、中间和结束部分连接起来
    # 最终结果 → [3.0, 5.0, 4.333, 5.667, 9.0]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    """
    训练基于策略的智能体（如 REINFORCE、A2C）
    每轮收集一条完整轨迹后更新智能体
    """
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    """
    训练基于经验回放的智能体（如 DQN、DDPG）
    每步将数据存入缓冲区，缓冲区足够大时采样小批量数据更新智能体
    """
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    """
    计算优势函数（Generalized Advantage Estimation, GAE）
    函数返回的是从第一个时间步到最后一个时间步的优势估计值。
    gamma: 折扣因子
    lmbda: GAE参数
    td_delta: 时序差分误差
    """
    # gamma = 0.9
    # lmbda = 0.8
    # td_delta = torch.tensor([0.1, 0.2, 0.3])    
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    # advantage_list = [0.3, 0.416, 0.39952]
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    # 反转结果列表 结果：[0.39952, 0.416, 0.3]
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
