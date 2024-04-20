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
            # print(state)
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
    y=[-0.316, -0.331, -0.27, -0.325, -0.27, -0.329, -0.212, -0.306, -0.309, -0.255, -0.329, -0.239, -0.337, 0.555, 0.275, 0.268, 0.246, 0.283, 0.2982, 0.2781, 0.1591, 0.0995, -0.0245, -0.0891, -0.1445, 0.0259, -0.0126, 0.2382, 0.179, 0.2034, 0.3675, 0.2764, 0.1596, 0.076, 0.0075, -0.0357, -0.0798, -0.1127, -0.1357, -0.1531, -0.1644, -0.1727, -0.1717, -0.1867, -0.1988, -0.2508, -0.2393, -0.1829, -0.1192, -0.0349]
    x=[0.0273, 0.0153, 0.01, 0.008, 0.006, 0.004, 0.002, 0.0075, 0.0, 0.031, 0.0059, 0.005, 0.0362, 0.0032, 0.0017, 0.0, 0.0001, 0.0001, 0.0001, 0.00028, 0.00087, 0.00135, 0.00176, 0.00214, 0.00251, 0.00288, 0.00329, 0.001, 0.00522, 0.00585, 0.00632, 0.00661]
    t=[]
    off=len(y)-len(x)
    for i in range(len(x)):
        t.append( [x[i],round(y[i+off]-y[i+off-1],5) ] )
    
            
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

    coefficients = np.polyfit(y, x, deg=3)

    # 解方程获取目标x值
    target_x = np.polyval(coefficients, -0.008142374622271024)
    print(target_x)
    # coefs = np.polyfit(x, y, 3)
    # poly = np.poly1d(coefs)
    # x_roots = np.roots(poly)
    # real_roots = x_roots[np.isreal(x_roots)].real
    # x_when_y_is_zero = real_roots[0]    
    # x_when_y_is_zero =min([v for v in real_roots if v>0])

    # print(real_roots)

if __name__ == '__main__':
    # run()
    test()