import os, time
import numpy as np
import torch

from model import PolicyNet
from agent import Agent, ACTIONS

GAME_ACTIONS_NUM = len(ACTIONS)
GAME_WIDTH, GAME_HEIGHT = 10, 20

ACTION_NAMES = {0: 'ROTATE', 1: 'LEFT', 2: 'RIGHT', 3: 'NONE', 4: 'DOWN'}


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
