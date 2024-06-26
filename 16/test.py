from agent_numba import Agent
from model import PolicyValueNet
import os,time
import numpy as np
import json

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
    t=[[0.00509, 0.024], [0.00402, 0.0068], [0.0037, 0.0029], [0.00356, -0.0164], [0.00447, 0.0119], [0.0039, -0.0015], [0.00402, 0.0056], [0.00374, -0.0002], [0.00377, 0.0003], [0.00377, 0.0117], [0.00312, -0.0078], [0.00354, -0.0181], [0.00448, 0.0021], [0.00443, 0.0297], [0.00289, -0.0148], [0.00365, -0.0062], [0.00397, -0.0013], [0.00405, 0.0175], [0.00318, -0.0012], [0.00323, -0.0136], [0.0039, -0.0065], [0.00422, 0.0158], [0.00347, 0.0119], [0.00287, -0.012], [0.00345, -0.014], [0.00411, -0.0042], [0.0043, 0.0312], [0.00287, -0.004], [0.00305, -0.0157], [0.00377, 0.0162], [0.00308, -0.0175], [0.00381, -0.0092], [0.00415, 0.0094], [0.00387, -0.0071], [0.00412, 0.0], [0.00416, 0.0266], [0.0033, -0.0012], [0.00334, -0.0268], [0.00416, -0.0396], [0.00558, 0.0178], [0.00493, 0.0088], [0.00461, -0.0076], [0.00483, -0.0035], [0.00485, -0.0007], [0.00485, 0.0076], [0.00474, -0.0106], [0.00396, 0.0028], [0.00388, -0.0216], [0.00175, -0.0286], [0.001, -0.0296]]
    
            
    # a={}
    # a["advantage"]=t
    # x = json.dumps(a)
    # print(x)
    # x = np.array(x)
    # y = np.array(y[-len(x):])
    # # # 添加一列全为1的常数列作为截距
    # # X = np.vstack([x, np.ones(len(x))]).T
    # # # 使用np.linalg.lstsq()进行线性回归
    # # coefficients, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # # # 提取回归系数
    # # slope = coefficients[0]
    # # intercept = coefficients[1]
    # # # 计算当y等于0时对应的x值
    # # x_when_y_is_zero = -intercept / slope
    x=[]
    y=[]
    for x1,y1 in t:
        x.append(x1)
        y.append(y1)
    
    # # 构建线性回归模型
    # X = np.vstack([x, np.ones(len(x))]).T
    # m, c = np.linalg.lstsq(X, y, rcond=None)[0]

    # # 反向求解x值
    # predicted_x = (-0.049 - c) / m
    # print(predicted_x)

    coefficients = np.polyfit(y, x, deg=2)

    # 解方程获取目标x值
    target_x = np.polyval(coefficients, -0.12584059779282147)
    print(target_x)
    # coefs = np.polyfit(x, y, 3)
    # poly = np.poly1d(coefs)
    # x_roots = np.roots(poly)
    # real_roots = x_roots[np.isreal(x_roots)].real
    # x_when_y_is_zero = real_roots[0]    
    # x_when_y_is_zero =min([v for v in real_roots if v>0])

    # print(real_roots)

if __name__ == '__main__':
    run()
    # test()