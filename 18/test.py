import os, time, json
import numpy as np
import torch

from model import PolicyNet
from agent import Agent, ACTIONS

GAME_ACTIONS_NUM = len(ACTIONS)
GAME_WIDTH, GAME_HEIGHT = 10, 20

ACTION_NAMES = {0: 'ROTATE', 1: 'LEFT', 2: 'RIGHT', 3: 'NONE', 4: 'DOWN'}


def plot_training_curves():
    """读取状态日志并在终端显示 PP_Piece 和 Te_Piece 曲线"""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    status_file = os.path.join(curr_dir, 'model', 'vit-ti', 'status.json')

    if not os.path.exists(status_file):
        print(f"状态文件不存在: {status_file}")
        return

    with open(status_file, 'r', encoding='utf-8') as f:
        status = json.load(f)

    history = status.get('history', [])
    if not history:
        print("历史记录为空")
        return

    # 提取数据
    pp_piece = [h.get('ppo_piececount', 0) for h in history]
    te_piece = [h.get('test_piececount', 0) for h in history]
    agents = [h.get('agent', 0) for h in history]

    # 绘制 ASCII 曲线
    width = 60
    height = 15

    def draw_curve(data, label):
        if not data:
            return
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            max_val = min_val + 1

        print(f"\n{label} (min={min_val:.1f}, max={max_val:.1f})")
        print('  ' + '─' * (width + 2))

        for row in range(height, -1, -1):
            threshold = min_val + (max_val - min_val) * row / height
            line = '  │'
            for col in range(width):
                idx = int(col * len(data) / width)
                if idx < len(data) and data[idx] >= threshold:
                    line += '█'
                else:
                    line += ' '
            val = min_val + (max_val - min_val) * row / height
            line += f'│ {val:.1f}'
            print(line)

        print('  ' + '─' * (width + 2))
        print(f'  Agent: {agents[0]} → {agents[-1]}')

    draw_curve(pp_piece, 'PP_Piece (训练)')
    draw_curve(te_piece, 'Te_Piece (测试)')

def play_one_game():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    mf = os.path.join(curr_dir, 'model', 'vit-ti', 'model.pth')
    policy_net = PolicyNet(GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=mf)

    agent = Agent(isRandomNextPiece=True)
    prev_action = 3  # KEY_NONE

    step = 0
    while not agent.terminal:
        state = np.array([agent.current_state()])
        state_tensor = torch.FloatTensor(state).to(policy_net.device)
        prev_action_tensor = torch.LongTensor([prev_action]).to(policy_net.device)

        policy_net.net.eval()
        with torch.no_grad():
            log_probs, values = policy_net.net(state_tensor, prev_action_tensor)
        probs = np.exp(log_probs[0].cpu().numpy())
        v_val = values[0].cpu().numpy()  # [N_q] quantiles
        v_median = v_val[len(v_val)//2]

        availables = agent.availables
        probs_masked = probs * availables.astype(np.float32)
        s = probs_masked.sum()
        if s > 1e-10:
            probs_masked = probs_masked / s
        else:
            probs_masked = availables.astype(np.float32)
            probs_masked = probs_masked / probs_masked.sum()

        action = int(np.argmax(probs_masked))

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"=== Step {step} === Action: {ACTION_NAMES[action]} | Piece: {agent.fallpiece['shape']} (rot={agent.fallpiece['rotation']}, x={agent.fallpiece['x']}, y={agent.fallpiece['y']})")
        agent.print()

        prev_action = action
        landed, removed = agent.step(action)
        print(f"Probs: {dict(zip(ACTION_NAMES.values(), [f'{p:.3f}' for p in probs_masked]))}")
        print(f"V(s) : {v_median:.3f} (quantiles: {v_val.round(3)})")
        # if landed:
        #     print(f"  >> LANDED  cleared={removed}  piececount={agent.piececount}")

        step += 1
        time.sleep(0.2)

    agent.print()
    print(f"\nGame Over! piececount={agent.piececount} removedlines={agent.removedlines} steps={agent.steps}")


if __name__ == '__main__':
    play_one_game()
    print("\n")
    plot_training_curves()
