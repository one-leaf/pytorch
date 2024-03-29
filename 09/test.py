from agent import Agent
from game import TetrominoEnv
from mcts import MCTSPlayer, MCTSPurePlayer
from model import PolicyValueNet
import time
import os


def run():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(curr_dir, './model/')
    model_file =  os.path.join(model_dir, 'model-cnn.pth')

    try:
        agent = Agent()    
        agent.limit_piece_count = 0
        agent.limit_max_height = 10
        # env = TetrominoEnv(agent.tetromino)    
        # 神经网络的价值策略
        net_policy = PolicyValueNet(10, 20, 5, model_file=model_file)
        mcts_ai_player = MCTSPlayer(net_policy.policy_value_fn, c_puct=1, n_playout=64)
        # agent.start_play(mcts_ai_player, env)
        while not agent.terminal:
            act = mcts_ai_player.get_action(agent)
            # agent.step(act, env)
            agent.step(act)
            print(agent.get_availables())
            agent.print2(True)
    except KeyboardInterrupt:
        print('quit')


if __name__ == '__main__':
    run()