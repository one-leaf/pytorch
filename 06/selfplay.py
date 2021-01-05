from shutil import copyfile
from torch import batch_norm, sqrt
from model import PolicyValueNet  
from mcts import MCTSPurePlayer, MCTSPlayer
from agent import Agent
import os, pickle, uuid
import logging
import numpy as np

import faulthandler
faulthandler.enable()

curr_dir = os.path.dirname(os.path.abspath(__file__))
size = 15  # 棋盘大小
n_in_row = 5  # 几子连线

data_dir = os.path.join(curr_dir, './data/%s_%s/'%(size,n_in_row))
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

data_wait_dir = os.path.join(curr_dir, './data/%s_%s_wait/'%(size,n_in_row))
if not os.path.exists(data_wait_dir):
    os.makedirs(data_wait_dir)

model_dir = os.path.join(curr_dir, './model/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_file =  os.path.join(model_dir, 'model_%s_%s.pth'%(size,n_in_row))
best_model_file =  os.path.join(model_dir, 'best_model_%s_%s.pth'%(size,n_in_row))

class FiveChessPlay():
    def __init__(self):
        self.policy_evaluate_size = 20  # 策略评估胜率时的模拟对局次数
        self.batch_size = 512  # 训练一批数据的长度
        self.max_keep_size = 500000  # 保留最近对战样本个数 平均一局大约400~600个样本, 也就是包含了最近1000次对局数据

        # 训练参数
        self.learn_rate = 1e-4
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # 概率缩放程度，实际预测0.01，训练采用1
        self.n_playout = 250  # 每个动作的模拟次数
        self.play_batch_size = 1 # 每次自学习次数
        self.epochs = 1  # 重复训练次数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        
        self.c_puct_win = [0, 0]

        # 纯MCTS的模拟数，用于评估策略模型
        self.pure_mcts_playout_num = 2000 # 用户纯MCTS构建初始树时的随机走子步数
        self.c_puct = 3  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5

        if os.path.exists(model_file):
            # 使用一个训练好的策略价值网络
            self.policy_value_net = PolicyValueNet(size, model_file=model_file)
        else:
            # 使用一个新的的策略价值网络
            self.policy_value_net = PolicyValueNet(size)

    def save_wait_data(self, obj):
        filename = "{}.pkl".format(uuid.uuid1())
        savefile = os.path.join(data_wait_dir, filename)
        pickle.dump(obj, open(savefile, "wb"))

    def get_equi_data(self, play_data):
        """
        通过旋转和翻转增加数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            mcts_porb = mcts_porb.reshape(size, size)
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                # equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(size, size)), i)
                equi_mcts_prob = np.rot90(mcts_porb, i)
                # extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                # extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
        return extend_data

    def collect_selfplay_data(self,i):
        """收集自我对抗数据用于训练"""       
        # 使用MCTS蒙特卡罗树搜索进行自我对抗
        logging.info("TRAIN Self Play starting ...")
        agent = Agent(size, n_in_row, is_shown=0)
        # 创建使用策略价值网络来指导树搜索和评估叶节点的MCTS玩家   
        if i%2==0:     
            mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)
        else:
            mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct+0.5, n_playout=self.n_playout, is_selfplay=1)

        mcts_player.mcts._limit_max_var=False

        # pure_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct+0.5, n_playout=self.n_playout, is_selfplay=1)

        # 开始下棋
        _, play_data = agent.start_self_play(mcts_player, None, temp=self.temp)
        agent.game.print()                   

        if i%2==0:
            self.c_puct_win[0] = self.c_puct_win[0]+1
        else:
            self.c_puct_win[1] = self.c_puct_win[1]+1

        play_data = list(play_data)[:]     
        # 采用翻转棋盘来增加样本数据集
        play_data = self.get_equi_data(play_data)
        logging.info("Self Play end. length:%s saving ..." % len(play_data))
        logging.info("c_puct:{}/{} = {}/{}".format(self.c_puct, self.c_puct+0.5, self.c_puct_win[0], self.c_puct_win[1]))

        # 保存训练数据
        for obj in play_data:
            self.save_wait_data(obj)
        return play_data[-1]

    def run(self):
        """启动训练,并动态调整c_puct参数"""
        try:
            for i in range(10000):
                logging.info("TRAIN Batch:{} starting, Size:{}, n_in_row:{}".format(i, size, n_in_row))
                state, mcts_porb, winner = self.collect_selfplay_data(i)

                if (i+1)%10 == 0:

                    if self.c_puct_win[0]>self.c_puct_win[1]:                               
                        self.c_puct=self.c_puct-0.1
                    if self.c_puct_win[0]<self.c_puct_win[1]:
                        self.c_puct=self.c_puct+0.1
                    self.c_puct = max(0.2, self.c_puct)
                    self.c_puct = min(10, self.c_puct)

                    self.policy_value_net = PolicyValueNet(size, model_file=model_file)
                    self.c_puct_win=[0, 0]

        except KeyboardInterrupt:
            logging.info('quit')

if __name__ == '__main__':
    # play
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    playing = FiveChessPlay()
    playing.run()