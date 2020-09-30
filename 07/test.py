from agent import Agent
from mcts import MCTSPlayer, MCTSPurePlayer
from model import PolicyValueNet
import time
import os


def run():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(curr_dir, './model/')
    model_file =  os.path.join(model_dir, 'model.pth')

    try:
        agent = Agent(need_draw=True)
        # 神经网络的价值策略
        net_policy = PolicyValueNet(100, 200, 4, model_file=model_file)
        mcts_ai_player = MCTSPlayer(net_policy.policy_value_fn, c_puct=5, n_playout=100)

        agent.start_play(mcts_ai_player)
    except KeyboardInterrupt:
        print('quit')


if __name__ == '__main__':
    run()