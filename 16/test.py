from agent_numba import Agent
from model import PolicyValueNet
import os,time
import numpy as np

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
            state = agent.current_state()
            print(state)
            act = net_policy.policy_value_fn_best_act(agent)
            # agent.step(act, env)
            agent.step(act)
            # print(agent.get_availables())
            # os.system("cls")
            # print(v, agent.position_to_action_name(act), (time.time()-start_time)/agent.steps,"s", flush=True)
            
            agent.print()
            time.sleep(0.2)
            
            # print(agent.current_state())
            input()
    except KeyboardInterrupt:
        print('quit')


def test():
    x = np.array([0.0273, 0.0153, 0.01, 0.008, 0.006, 0.004, 0.002, 0.0075, 0.0, 0.031, 0.0059, 0.005])
    y = np.array([0.2982, 0.2781, 0.1591, 0.0995, -0.0245, -0.0891, -0.1445, 0.0259, -0.0126, 0.2382, 0.179, 0.2034])

    # # 添加一列全为1的常数列作为截距
    # X = np.vstack([x, np.ones(len(x))]).T
    # # 使用np.linalg.lstsq()进行线性回归
    # coefficients, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # # 提取回归系数
    # slope = coefficients[0]
    # intercept = coefficients[1]
    # # 计算当y等于0时对应的x值
    # x_when_y_is_zero = -intercept / slope
    
    coefs = np.polyfit(x, y, 3)
    poly = np.poly1d(coefs)
    x_roots = np.roots(poly)
    real_roots = x_roots[np.isreal(x_roots)].real
    x_when_y_is_zero = real_roots[0]    
    x_when_y_is_zero =min([v for v in real_roots if v>0])

    print(x_when_y_is_zero)

if __name__ == '__main__':
    # run()
    test()