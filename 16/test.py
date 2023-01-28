from agent import Agent, ACTIONS
from model import PolicyValueNet
from mcts_single import MCTSPlayer
import os,time


def run():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(curr_dir, './model/')
    model_file =  os.path.join(model_dir, 'vit-ti/model.pth')

    try:
        agent = Agent(isRandomNextPiece=True) 
        agent.show_mcts_process = True   
        # agent.limit_piece_count = 0
        # agent.limit_max_height = 10
        # env = TetrominoEnv(agent.tetromino)    
        # 神经网络的价值策略
        net_policy = PolicyValueNet(10, 20, 4, model_file=model_file)
        mcts_ai_player = MCTSPlayer(net_policy.policy_value_fn, c_puct=5, n_playout=100)
        # mcts_ai_player.mcts.ext_reward = False
        # agent.start_play(mcts_ai_player, env)
        start_time = time.time()
        while not agent.terminal:
            # act_probs, v = net_policy.policy_value_fn(agent)
            # act, act_p = 0, 0
            # for a, p in act_probs:
            #     if p > act_p:
            #         act, act_p = a, p
            act, act_probs, v, qval, acc_ps, depth, state_n = mcts_ai_player.get_action(agent, 0)
            # agent.step(act, env)

            agent.step(act)
            # print(agent.get_availables())
            # os.system("cls")
            print(v, agent.position_to_action_name(act), (time.time()-start_time)/agent.steps,"s")
            agent.print2()
            time.sleep(0.1)
            # print(agent.current_state())
            # input()
    except KeyboardInterrupt:
        print('quit')


if __name__ == '__main__':
    run()