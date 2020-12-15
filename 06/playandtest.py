from shutil import copyfile
from torch import batch_norm, sqrt
from model import PolicyValueNet  
from mcts import MCTSPurePlayer, MCTSPlayer
from agent import Agent
import os, glob, pickle, uuid
import sys, time
import random
import logging
import numpy as np
from collections import defaultdict, deque
import torch
from threading import Thread, Lock

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
        self.n_playout = 1000  # 每个动作的模拟次数
        self.play_batch_size = 1 # 每次自学习次数
        self.epochs = 1  # 重复训练次数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标       

        # 纯MCTS的模拟数，用于评估策略模型
        self.pure_mcts_playout_num = 500 # 用户纯MCTS构建初始树时的随机走子步数
        self.c_puct = 1.5  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5
        self.mcts_win = [0, 0]


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

    def collect_selfplay_data(self):
        """收集自我对抗数据用于训练"""       
        # 使用MCTS蒙特卡罗树搜索进行自我对抗
        logging.info("TRAIN Self Play starting ...")
        agent = Agent(size, n_in_row, is_shown=0)
        # 创建使用策略价值网络来指导树搜索和评估叶节点的MCTS玩家
        mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

        # 有一定几率和纯MCTS对抗
        r = random.random()
        if r>0.5:
            pure_mcts_player = MCTSPurePlayer(c_puct=self.c_puct, n_playout=self.pure_mcts_playout_num)
            print("AI VS MCTS, pure_mcts_playout_num:", self.pure_mcts_playout_num)
        else:
            pure_mcts_player = None

        # 开始下棋
        winner, play_data = agent.start_self_play(mcts_player, pure_mcts_player, temp=self.temp)
        agent.game.print()                   

        if not pure_mcts_player is None:
            if winner == mcts_player.player:
                self.mcts_win[0] = self.c_puct_win[0]+1
            if winner == pure_mcts_player.player:
                self.mcts_win[1] = self.c_puct_win[1]+1
            logging.info("pure_mcts_playout_num:{} = {}/{}".format(self.pure_mcts_playout_num, self.mcts_win[0], self.mcts_win[1]))

        play_data = list(play_data)[:]     
        # 采用翻转棋盘来增加样本数据集
        play_data = self.get_equi_data(play_data)
        logging.info("Self Play end. length:%s saving ..." % len(play_data))

        # 保存训练数据
        for obj in play_data:
            self.save_wait_data(obj)
        return play_data[-1]

    
    def policy_evaluate(self, n_games=10):
        """
        策略胜率评估：当前模型与最佳模型对战n局看胜率
        """
        # 当前训练好的模型
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        # 如果不存在最佳模型，直接将当前模型保存为最佳模型
        if not os.path.exists(best_model_file):
            self.policy_value_net.save_model(best_model_file)
            return

        best_policy_value_net = PolicyValueNet(size, model_file=best_model_file)
        best_mcts_player = MCTSPlayer(best_policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        win_cnt = defaultdict(int)
        for i in range(n_games):  # 对战
            agent = Agent(size, n_in_row, is_shown=0)
            winner, play_data = agent.start_self_evaluate(current_mcts_player, best_mcts_player, temp=self.temp, start_player=i % 2)
            if winner == current_mcts_player.player:
                win_cnt[0] += 1  # 赢
                print("Curr Model Win!","win:", win_cnt[0],"lost",win_cnt[1],"tie",win_cnt[-1])
            elif winner == -1:  
                win_cnt[-1] += 1 # 平局
                print("Tie!","win:", win_cnt[0],"lost",win_cnt[1],"tie",win_cnt[-1])
            else:
                win_cnt[1] += 1  # 输
                print("Curr Model Lost!","win:", win_cnt[0],"lost",win_cnt[1],"tie",win_cnt[-1])
                
                # 如果输了就保存训练数据
                play_data = list(play_data)[:]
                play_data = self.get_equi_data(play_data)
                logging.info("Eval Play end. length:%s saving ..." % len(play_data))
                for obj in play_data:
                    self.save_wait_data(obj)

            agent.game.print()

        win_ratio = win_cnt[0] / n_games

        logging.info("curr model vs best model: win: {}, lose: {}, tie: {}, win_ratio: {}".format(
            win_cnt[0], win_cnt[1], win_cnt[-1], win_ratio))

        # 如果当前模型的胜率大于等于0.6,保留为最佳模型
        if win_ratio>=0.6:
            t = os.path.getctime(best_model_file)
            timeStruct = time.localtime(t)
            timestr = time.strftime('%Y_%m_%d_%H_%M', timeStruct)
            os.rename(best_model_file, best_model_file+"."+timestr)
            self.policy_value_net.save_model(best_model_file)
            print("save curr modle to best model")

        return win_ratio

    def run(self):
        """启动训练"""
        try:
            # 先训练样本10000局
            for i in range(10000):
                logging.info("TRAIN Batch:{} starting, Size:{}, n_in_row:{}".format(i, size, n_in_row))
                state, mcts_porb, winner = self.collect_selfplay_data()
                if i == 0: 
                    print("-"*50,"state","-"*50)
                    print(state)
                    print("-"*50,"mcts_porb","-"*50)
                    print(mcts_porb)
                    print("-"*50,"winner","-"*50)
                    print(winner)

                if (i+1)%10 == 0:
                    self.policy_evaluate(self.policy_evaluate_size)
    
                    if self.mcts_win[0]>self.mcts_win[1]:                               
                        self.pure_mcts_playout_num=self.pure_mcts_playout_num-10
                    if self.mcts_win[0]<self.mcts_win[1]:
                        self.pure_mcts_playout_num=self.pure_mcts_playout_num+10
                    if self.pure_mcts_playout_num<100: self.pure_mcts_playout_num=100
                    if self.pure_mcts_playout_num>5000: self.pure_mcts_playout_num=5000
                    
                    self.mcts_win=[0, 0]
                    self.policy_value_net = PolicyValueNet(size, model_file=model_file)


            # 一轮训练完毕后与最佳模型进行对比
            # # 如果输了，再训练一次
            # if win_ratio<=0.5:
            #     self.policy_evaluate(self.policy_evaluate_size)
            #     print("lost all, add more sample")
        except KeyboardInterrupt:
            logging.info('quit')

if __name__ == '__main__':
    # play
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    playing = FiveChessPlay()
    playing.run()