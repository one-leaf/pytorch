import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob

# 如果有显卡就用显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':

    task = 'landing'  # 悬停 'hover' 或 着陆 'landing'
    max_steps = 800
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = sorted(glob.glob(os.path.join(curr_dir, task+'_ckpt', '*.pt')))[-1]  # 最新的模型

    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if os.path.exists(ckpt_dir):
        print("load checkpoint", ckpt_dir)
        checkpoint = torch.load(ckpt_dir,map_location=device)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    state = env.reset()
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render(window_name='test')
        if env.already_crash:
            break

