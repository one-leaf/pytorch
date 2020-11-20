from shutil import copyfile
from torch import batch_norm, sqrt
from model import PolicyValueNet  
from mcts import MCTSPurePlayer, MCTSPlayer
from agent import Agent
import os, glob, pickle
import sys
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

model_dir = os.path.join(curr_dir, './model/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_file =  os.path.join(model_dir, 'model_%s_%s.pth'%(size,n_in_row))
best_model_file =  os.path.join(model_dir, 'best_model_%s_%s.pth'%(size,n_in_row))

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    # trans_count 希望训练的总记录数，为 训练轮次 * Batch_Size
    # max_keep_size 最多保存的训练样本数
    def __init__(self, data_dir, max_keep_size):
        self.data_dir = data_dir
        self.max_keep_size = max_keep_size
        self.index = 0
        self.data_index_file = os.path.join(data_dir, 'index.txt')
        self.file_list = deque(maxlen=max_keep_size)        
        self._save_lock = Lock()
        self.load_index()
        self.load_game_files()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        state, mcts_prob, winner = pickle.load(open(filename, "rb"))
        state = torch.from_numpy(state).float()
        mcts_prob = torch.from_numpy(mcts_prob).float()
        winner = torch.as_tensor(winner).float()
        return state, mcts_prob, winner

    def load_game_files(self):
        files = glob.glob(os.path.join(self.data_dir, "*.pkl"))
        files = sorted(files, key=lambda x: os.path.getmtime(x))
        for filename in files:
            self.file_list.append(filename)

    def save_index(self):
        with open(self.data_index_file, "w") as f:
            f.write(str(self.index))

    def load_index(self):
        if os.path.exists(self.data_index_file):
            self.index = int(open(self.data_index_file, 'r').read().strip())

    def save(self, obj):
        with self._save_lock:
            filename = "{}.pkl".format(self.index % self.max_keep_size,)
            savefile = os.path.join(self.data_dir, filename)
            pickle.dump(obj, open(savefile, "wb"))
            self.index += 1
            self.save_index()

class FiveChessTrain():
    def __init__(self):
        self.policy_evaluate_size = 10  # 策略评估胜率时的模拟对局次数
        self.batch_size = 512  # 训练一批数据的长度
        self.max_keep_size = 300000  # 保留最近对战样本个数 平均一局大约400~600个样本, 也就是包含了最近800次对局数据

        # 训练参数
        self.learn_rate = 1e-5
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1  # 概率缩放程度，实际预测0.01，训练采用1
        self.n_playout = 500  # 每个动作的模拟次数
        self.play_batch_size = 1 # 每次自学习次数
        self.epochs = 5  # 重复训练次数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        
        # 纯MCTS的模拟数，用于评估策略模型
        self.pure_mcts_playout_num = 4000 # 用户纯MCTS构建初始树时的随机走子步数
        self.c_puct = 5  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度 默认 5

        if os.path.exists(model_file):
            # 使用一个训练好的策略价值网络
            self.policy_value_net = PolicyValueNet(size, model_file=model_file)
        else:
            # 使用一个新的的策略价值网络
            self.policy_value_net = PolicyValueNet(size)

        print("start data loader")
        self.dataset = Dataset(data_dir, self.max_keep_size)
        print("dataset len:",len(self.dataset),"index:",self.dataset.index)
        print("end data loader")

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
        if random.random()>0.99:
            pure_mcts_player = MCTSPurePlayer(c_puct=5, n_playout=2000)
            print("AI VS MCTS, pure_mcts_playout_num:", 2000)
        else:
            pure_mcts_player = None

        # 开始下棋
        winner, play_data = agent.start_self_play(mcts_player, pure_mcts_player, temp=self.temp)
        agent.game.print()                   

        play_data = list(play_data)[:]     
        # 如果数据不够，就采用翻转棋盘来增加样本数据集
        if len(self.dataset)<self.max_keep_size/2:
            play_data = self.get_equi_data(play_data)
        logging.info("Self Play end. length:%s saving ..." % len(play_data))

        # 保存训练数据
        for obj in play_data:
            self.dataset.save(obj)
        return play_data[-1]

    def policy_update(self, sample_data, epochs=1):
        """更新策略价值网络policy-value"""
        # 训练策略价值网络
        state_batch, mcts_probs_batch, winner_batch = sample_data

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)  
        for i in range(epochs):
            loss, v_loss, p_loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            # 散度计算：
            # D(P||Q) = sum( pi * log( pi / qi) ) = sum( pi * (log(pi) - log(qi)) )
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * epochs:  # 如果D_KL跑偏则尽早停止
                break

        # 自动调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # 如果学习到了，explained_var 应该趋近于 1，如果没有学习到也就是胜率都为很小值时，则为 0
        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        # entropy 信息熵，越小越好
        logging.info(("TRAIN kl:{:.5f},lr_multiplier:{:.3f},v_loss:{:.5f},p_loss:{:.5f},entropy:{:.5f},var_old:{:.5f},var_new:{:.5f}"
                      ).format(kl, self.lr_multiplier, v_loss, p_loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy

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
            winner, play_data = agent.start_self_evaluate(current_mcts_player, best_mcts_player, temp=1, start_player=i % 2)
            if winner == current_mcts_player.player:
                win_cnt[0] += 1  # 赢
                print("Curr Model Win!","win:", win_cnt[0],"lost",win_cnt[1],"tie",win_cnt[-1])
            elif winner == -1:  
                win_cnt[-1] += 1 # 平局
                print("Tie!","win:", win_cnt[0],"lost",win_cnt[1],"tie",win_cnt[-1])
            else:
                win_cnt[1] += 1  # 输
                print("Curr Model Lost!","win:", win_cnt[0],"lost",win_cnt[1],"tie",win_cnt[-1])
            
            agent.game.print()

            # 保存训练数据
            play_data = list(play_data)[:]
            # 如果训练数据不够，就通过翻转增加样本
            if len(self.dataset)<self.max_keep_size/2:
                play_data = self.get_equi_data(play_data)
            logging.info("Eval Play end. length:%s saving ..." % len(play_data))
            for obj in play_data:
                self.dataset.save(obj)

        win_ratio = win_cnt[0] / n_games

        logging.info("curr model vs best model: win: {}, lose: {}, tie: {}, win_ratio: {}".format(
            win_cnt[0], win_cnt[1], win_cnt[-1], win_ratio))

        # 如果当前模型的胜率大于等于0.7,保留为最佳模型
        if win_ratio>=0.7:
            self.policy_value_net.save_model(best_model_file)
            print("save curr modle to best model")

        return win_ratio

    def run(self):
        """启动训练"""
        try:
            # 早期补齐训练样本
            step = 0
            if len(self.dataset)/self.max_keep_size<0.1:
                for _ in range(8):
                    logging.info("TRAIN Batch:{} starting, Size:{}, n_in_row:{}".format(step + 1, size, n_in_row))
                    state, mcts_porb, winner = self.collect_selfplay_data()
                    print("-"*50,"state","-"*50)
                    print(state)
                    print("-"*50,"mcts_porb","-"*50)
                    print(mcts_porb)
                    print("-"*50,"winner","-"*50)
                    print(winner)
                    logging.info("TRAIN Batch:{} end".format(step + 1,))
                    step += 1
                if len(self.dataset)<self.batch_size*10:
                    return                

            training_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,)
            tran_epochs = int(len(self.dataset)/(self.batch_size))
            for i, data in enumerate(training_loader):  # 计划训练批次
                # 使用对抗数据训练策略价值网络模型
                loss, entropy = self.policy_update(data, self.epochs)
               
                # 训练中间插入20局自我对战样本
                if (i+1) % (tran_epochs//20) == 0:
                    self.policy_value_net.save_model(model_file)
                    # 收集自我对抗数据
                    for _ in range(self.play_batch_size):
                        self.collect_selfplay_data()
                    logging.info("TRAIN self-play end, {} / {}".format(i*self.batch_size, len(self.dataset)))
                   
            # 一轮训练完毕后与最佳模型进行对比
            win_ratio = self.policy_evaluate(self.policy_evaluate_size)
            # 如果输了，再训练一次
            if win_ratio<=0.5:
                self.policy_evaluate(self.policy_evaluate_size)
                print("lost all, add more sample")
        except KeyboardInterrupt:
            logging.info('quit')

if __name__ == '__main__':
    # train
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    training = FiveChessTrain()
    # training.policy_evaluate(training.policy_evaluate_size)
    training.run()