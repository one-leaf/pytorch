from torch import sqrt
from policy_value_net import PolicyValueNet  
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, game_batch_num, buffer_size):
        self.data_dir = data_dir
        self.game_batch_num = game_batch_num
        self.buffer_size = buffer_size
        self.curr_game_batch_num = 0
        self.data_index_file = os.path.join(data_dir, 'index.txt')
        self.file_list = deque(maxlen=buffer_size)        
        self.load_game_batch_num()
        self.load_game_files()

    def __len__(self):
        return self.game_batch_num

    def __getitem__(self, index):
        filename = random.choice(self.file_list)
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

    def save_game_batch_num(self):
        with open(self.data_index_file, "w") as f:
            f.write(str(self.curr_game_batch_num))

    def load_game_batch_num(self):
        if os.path.exists(self.data_index_file):
            self.curr_game_batch_num = int(open(self.data_index_file, 'r').read().strip())

    def save(self, obj):
        filename = "%s.pkl" % self.curr_game_batch_num
        savefile = os.path.join(self.data_dir, filename)
        pickle.dump(obj, open(savefile, "wb"))
        self.curr_game_batch_num += 1
        self.save_game_batch_num()
        self.file_list.append(savefile)

    def curr_size(self):
        return len(self.file_list)

class FiveChessTrain():
    def __init__(self):
        self.policy_evaluate_size = 10  # 策略评估胜率时的模拟对局次数
        self.game_batch_num = 10000  # selfplay对战次数
        self.batch_size = 512  # data_buffer中对战次数超过n次后开始启动模型训练
        self.check_freq = 100  # 每对战n次检查一次当前模型vs旧模型胜率        

        # training params
        self.learn_rate = 1e-5
        self.lr_multiplier = 1.0  # 基于KL的自适应学习率
        self.temp = 1.0  # the temperature param
        self.n_playout = 800  # 每个动作的模拟次数
        self.buffer_size = 100000  # cache对战记录个数
        self.play_batch_size = 4 # 每次自学习次数
        self.epochs = 2  # 每次更新策略价值网络的训练步骤数, 推荐是5
        self.kl_targ = 0.02  # 策略价值网络KL值目标
        self.best_win_ratio = 0.0
        
        # 纯MCTS的模拟数，用于评估策略模型
        self.pure_mcts_playout_num = 1000 # 用户纯MCTS构建初始树时的随机走子步数
        self.c_puct = 3  # MCTS child权重， 用来调节MCTS中 探索/乐观 的程度
        if os.path.exists(model_file):
            # 使用一个训练好的策略价值网络
            self.policy_value_net = PolicyValueNet(size, model_file=model_file)
        else:
            # 使用一个新的的策略价值网络
            self.policy_value_net = PolicyValueNet(size)

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

    def collect_selfplay_data(self, lock):
        """收集自我对抗数据用于训练"""       
        # 使用MCTS蒙特卡罗树搜索进行自我对抗
        logging.info("TRAIN Self Play starting ...")
        agent = Agent(size, n_in_row, is_shown=0)
        # 创建使用策略价值网络来指导树搜索和评估叶节点的MCTS玩家
        mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)
        # 开始下棋
        winner, play_data = agent.start_self_play(mcts_player, temp=self.temp)
        play_data = list(play_data)[:]
        episode_len = len(play_data)
        # 把翻转棋盘数据加到数据集里
        play_data = self.get_equi_data(play_data)
        logging.info("TRAIN Self Play end. length:%s saving ..." % episode_len)

        # 保存对抗数据到data_buffer
        with lock:
            for obj in play_data:
                self.dataset.save(obj)
        agent.game.print(play_data[-1][0])                   

    def policy_update(self, sample_data, epochs=1):
        """更新策略价值网络policy-value"""
        # 训练策略价值网络
        # 随机抽取data_buffer中的对抗数据
        # mini_batch = self.dataset.loadData(sample_data)
        state_batch, mcts_probs_batch, winner_batch = sample_data
        # # for x in mini_batch:
        # #     print("-----------------")
        # #     print(x)
        # # state_batch = [data[0] for data in mini_batch]
        # # mcts_probs_batch = [data[1] for data in mini_batch]
        # # winner_batch = [data[2] for data in mini_batch]

        # print(state_batch)

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)  
        for i in range(epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            # 散度计算：
            # D(P||Q) = sum( pi * log( pi / qi) ) = sum( pi * (log(pi) - log(qi)) )
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # 如果D_KL跑偏则尽早停止
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
        logging.info(("TRAIN kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{:.5f},var_old:{:.5f},var_new:{:.5f}"
                      ).format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        策略胜率评估：模型与纯MCTS玩家对战n局看胜率
        """
        # AlphaGo Zero风格的MCTS玩家（使用策略价值网络来指导树搜索和评估叶节点）
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        # 纯MCTS玩家
        pure_mcts_player = MCTSPurePlayer(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):  # 对战
            agent = Agent(size, n_in_row, is_shown=0)
            winner = agent.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2)
            if winner == current_mcts_player.player:
                win_cnt[0] += 1
                print("AI Win!","win:", win_cnt[0],"lost",win_cnt[1],"tie",win_cnt[-1])
            elif winner == -1:  # 平局
                win_cnt[-1] += 1
                print("Tie!","win:", win_cnt[0],"lost",win_cnt[1],"tie",win_cnt[-1])
            else:
                win_cnt[1] += 1
                print("AI Lost!","win:", win_cnt[0],"lost",win_cnt[1],"tie",win_cnt[-1])
            
            agent.game.print()
        # MCTS的胜率 winner = 0, 1, -1 ; -1 表示平局
        win_ratio = 1.0 * (win_cnt[0] + 0.5 * win_cnt[-1]) / n_games
        logging.info("TRAIN Num_playouts: {}, win: {}, lose: {}, tie: {}, win_ratio: {}".format(self.pure_mcts_playout_num,
                                                                               win_cnt[0], win_cnt[1], win_cnt[-1], win_ratio))
        return win_ratio

    def run(self):
        """启动训练"""
        try:
            print("start data loader")
            self.dataset = Dataset(data_dir, self.game_batch_num*self.batch_size, self.buffer_size)
            training_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4,)
            print("end data loader")

            step = 0
            while self.dataset.curr_size() < self.batch_size*self.epochs:
                logging.info("TRAIN Batch:{} starting, Size:{}, n_in_row:{}".format(step + 1, size, n_in_row))
                lock = Lock()
                self.collect_selfplay_data(lock)
                logging.info("TRAIN Batch:{} end".format(step + 1,))
                step += 1

            for i, data in enumerate(training_loader):  # 计划训练批次
                # 使用对抗数据重新训练策略价值网络模型
                loss, entropy = self.policy_update(data, self.epochs)
                # 每n个batch检查一下当前模型胜率

                if (step + 1) % self.check_freq == 0:
                    # 保存buffer数据
                    logging.info("TRAIN Current self-play batch: {}".format(step + 1))
                    # 策略胜率评估：模型与纯MCTS玩家对战n局看胜率
                    win_ratio = self.policy_evaluate(self.policy_evaluate_size)
                    if win_ratio > self.best_win_ratio:  # 胜率超过历史最优模型
                        logging.info("TRAIN New best policy!!!!!!!!batch:{} win_ratio:{}->{} pure_mcts_playout_num:{}".format(step + 1, self.best_win_ratio, win_ratio, self.pure_mcts_playout_num))
                        self.best_win_ratio = win_ratio
                        # 保存当前模型为最优模型best_policy
                        self.policy_value_net.save_model(best_model_file)
                        # 如果胜率=100%，则增加纯MCT的模拟数 (<6000的限制视mem情况)
                        if self.best_win_ratio == 1.0: # and self.pure_mcts_playout_num < 6000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

                if (i+1) % (int(self.dataset.curr_size() ** 0.1)) == 0:
                    self.policy_value_net.save_model(model_file)
                    # 收集自我对抗数据

                    p_list=[]
                    lock = Lock()
                    for _ in range(self.play_batch_size):
                        p = Thread(target=self.collect_selfplay_data, args=(lock,))
                        p_list.append(p)
                        p.start()   

                    for p in p_list:
                        p.join()   
                    step += 1

                    # self.collect_selfplay_data(self.play_batch_size)
    
        except KeyboardInterrupt:
            logging.info('quit')


if __name__ == '__main__':
    # train
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    training = FiveChessTrain()
    training.run()