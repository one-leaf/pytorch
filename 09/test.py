from agent import Agent
from game import TetrominoEnv
from mcts import MCTSPlayer, MCTSPurePlayer
from model import PolicyValueNet
import time
import os


def run():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(curr_dir, './model/')
    model_file =  os.path.join(model_dir, 'model.pth')

    try:
        agent = Agent()    
        agent.limit_piece_count = 0
        env = TetrominoEnv(agent.tetromino)    
        # 神经网络的价值策略
        net_policy = PolicyValueNet(10, 20, 5, model_file=model_file)
        mcts_ai_player = MCTSPlayer(net_policy.policy_value_fn, c_puct=1, n_playout=64)
        # agent.start_play(mcts_ai_player, env)
        while not agent.terminal:
            if agent.curr_player == 0:
                # act_probs, value = net_policy.policy_value_fn(agent)
                # act = max(act_probs,  key=lambda act_prob: act_prob[1])[0]
                # print(act, act_probs, value)
                act = mcts_ai_player.get_action(agent)
            else:
                act = 4
            agent.step(act, env)
            
        agent.print()
    except KeyboardInterrupt:
        print('quit')


if __name__ == '__main__':
    run()