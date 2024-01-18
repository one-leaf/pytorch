from agent_numba import Agent
from model import PolicyValueNet
import os,time


def run():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(curr_dir, './model/')
    model_file =  os.path.join(model_dir, 'vit-ti/model.pth')

    try:
        agent = Agent(isRandomNextPiece=True) 
        agent.show_mcts_process = True   
        # env = TetrominoEnv(agent.tetromino)    
        # 神经网络的价值策略
        net_policy = PolicyValueNet(10, 20, 4, model_file=model_file)
        # agent.start_play(mcts_ai_player, env)
        start_time = time.time()
        while not agent.terminal:
            act = net_policy.policy_value_fn_best_act(agent)
            # agent.step(act, env)
            agent.step(act)
            # print(agent.get_availables())
            # os.system("cls")
            # print(v, agent.position_to_action_name(act), (time.time()-start_time)/agent.steps,"s", flush=True)
            
            agent.print()
            time.sleep(0.2)
            
            # print(agent.current_state())
            # input()
    except KeyboardInterrupt:
        print('quit')


if __name__ == '__main__':
    run()