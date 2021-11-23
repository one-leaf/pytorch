import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as pltsud
from matplotlib import pyplot as plt
import utils
import os
import glob

# 如果有显卡就用显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':

    task = 'landing'  # 悬停 'hover' 或 着陆 'landing'

    # 最大局数
    max_m_episode = 800000
    # 每局最大步数
    max_steps = 800

    # 建立一个火箭对象
    env = Rocket(task=task, max_steps=max_steps)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_folder = os.path.join(curr_dir, task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []

    # 创建策略网络
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # 加载最后一个模型
        last_ckpt = sorted(glob.glob(os.path.join(ckpt_folder, '*.pt')))[-1]
        print("load checkpoint", last_ckpt)
        checkpoint = torch.load(last_ckpt,map_location=device)
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']

    for episode_id in range(last_episode_id, max_m_episode):

        # 循环一局训练
        state = env.reset()
        rewards, log_probs, values, masks = [], [], [], []
        for step_id in range(max_steps):
            # 根据策略网络计算动作和对数概率，并获取动作值 
            action, log_prob, value = net.get_action(state)
            # 执行动作并获取下一个状态和奖励
            state, reward, done = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)
            
            if episode_id % 1000 == 1:
                env.render()

            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(state)
                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))

        # 保存模型
        if episode_id % 1000 == 1:
            plt.figure()
            plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id': episode_id,
                        'REWARDS': REWARDS,
                        'model_G_state_dict': net.state_dict()},
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))



