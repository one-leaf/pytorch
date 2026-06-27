import os, glob, pickle

from model import PolicyValueNet, data_dir, data_wait_dir, model_file
from agent import Agent, ACTIONS
from selfplay import GRPOSelfPlay

import time
from datetime import datetime
import os, math, copy

import numpy as np
import torch

from status import save_status_file, read_status_file, set_status_value

# 定义游戏的动作
GAME_ACTIONS_NUM = len(ACTIONS)
GAME_WIDTH, GAME_HEIGHT = 10, 20


class GRPODataset(torch.utils.data.Dataset):
    """GRPO 数据集，读取 pickle 格式: (state, ref_prob, advantage, action, mask)"""
    def __init__(self, data_dir, max_keep_size, test_size, epochs=5):
        self.data_dir = data_dir
        self.max_keep_size = max_keep_size
        self.test_size = test_size
        self.epoch = epochs
        self.file_list = []
        self.newsample = []
        self.data = {}
        self._flat_index = []
        self._test_flat_index = []
        self.copy_wait_file()
        self.load_game_files()
        self.calc_data()
        self.test = False

    def __len__(self):
        return len(self._flat_index) if not self.test else len(self._test_flat_index)

    def __getitem__(self, index):
        fn, step_idx = (self._flat_index if not self.test else self._test_flat_index)[index]
        state, ref_prob, advantage, action, mask = self.data[fn][step_idx]
        state = torch.from_numpy(state).float()
        ref_prob = torch.from_numpy(ref_prob).float()
        advantage = torch.as_tensor(advantage).float()
        action = torch.as_tensor(action).long()
        mask = torch.as_tensor(mask).long()
        return state, ref_prob, advantage, action, mask

    def load_game_files(self):
        print("start load files name ... ")
        start_time = time.time()
        files = glob.glob(os.path.join(self.data_dir, "*.pkl"))
        files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)

        if len(files) == 0:
            print("no data files found")
            return

        modified_time = os.path.getmtime(files[-1])
        self.first_time = time.localtime(modified_time)
        print("first time:", time.strftime('%y-%m-%d %H:%M:%S', self.first_time))
        modified_time = os.path.getmtime(files[0])
        self.last_time = time.localtime(modified_time)
        print("last time:", time.strftime('%y-%m-%d %H:%M:%S', self.last_time))

        delcount = 0
        for i, filename in enumerate(files):
            if i >= self.max_keep_size or os.path.getsize(filename) == 0:
                os.remove(filename)
                delcount += 1
            else:
                self.file_list.append(filename)

        pay_time = round(time.time() - start_time, 2)
        print("loaded data, total:", len(self.file_list), "delete:", delcount, "paid time:", pay_time)

    def calc_data(self):
        print("start load data to memory ...")
        start_time = time.time()
        for i, fn in enumerate(self.file_list):
            try:
                with open(fn, "rb") as f:
                    steps = pickle.load(f)

                for step in steps:
                    state, ref_prob, advantage, action, mask = step
                    assert state.shape == (4, 20, 10), f'error: state shape {state.shape}'
                    assert ref_prob.shape == (5,), f'error: ref_prob shape {ref_prob.shape}'
                    assert not np.isnan(advantage), f'error: advantage is Nan'
                    assert not np.isinf(advantage), f'error: advantage is Inf'

                self.data[fn] = steps

            except Exception as e:
                print(f"filename {fn} error can't load: {e}")
                if os.path.exists(fn):
                    os.remove(fn)
                if fn in self.file_list:
                    self.file_list.remove(fn)
                continue

        # advantage 已在 selfplay 中组内正规化，此处不再归一化
        advs = np.array([step[2] for file_steps in self.data.values() for step in file_steps])
        print(f"Advantage stats: min={advs.min():.3f} mean={advs.mean():.3f} max={advs.max():.3f} std={advs.std():.3f}")

        # 构建 flat index: [(filename, step_idx), ...]
        self._flat_index = [(fn, i) for fn in self.file_list for i in range(len(self.data[fn]))]
        self._test_flat_index = [(fn, i) for fn in self.newsample for i in range(len(self.data.get(fn, [])))]

        pay_time = round(time.time() - start_time, 2)
        print("loaded to memory, paid time:", pay_time)
        print("load data end")

    def copy_wait_file(self):
        print("start copy wait file to train ...")
        files = glob.glob(os.path.join(data_wait_dir, "*.pkl"))
        movefiles = sorted(files, key=lambda x: os.path.getmtime(x))
        time.sleep(1)

        i = -1
        movefiles_count = self.max_keep_size // self.epoch
        if len(movefiles) < movefiles_count:
            print(f"Insufficient data: have {len(movefiles)}, need {movefiles_count}")
            raise Exception("NEED SOME NEW DATA TO TRAIN")

        for i, fn in enumerate(movefiles):
            filename = os.path.basename(fn)
            savefile = os.path.join(self.data_dir, filename)
            if os.path.exists(savefile):
                os.remove(savefile)
            os.rename(fn, savefile)
            if self.test_size == -1 or len(self.newsample) < self.test_size:
                self.newsample.append(savefile)
            if (i + 1) >= movefiles_count and len(movefiles) - i <= 2 * movefiles_count:
                break

        print(f"mv {i + 1}/{len(movefiles)} files to train")

    def curr_size(self):
        return len(self.file_list)


class GRPOTestDataset(torch.utils.data.Dataset):
    """测试数据集：共享父数据集的 data，通过 test=True 切换 flat index"""
    def __init__(self, parent: GRPODataset):
        self.parent = parent
        parent.test = True

    def __len__(self):
        return len(self.parent._test_flat_index)

    def __getitem__(self, index):
        return self.parent.__getitem__(index)

class GRPOTrain():
    def __init__(self):
        self.batch_size = 32            # 每批训练样本数
        self.learn_rate = 1e-5
        self.lr_multiplier = 1.0
        self.buffer_size = 204800       # 最大保存数据量
        self.epochs = 50                # 每次更新的训练步骤数
        self.kl_targ = 0.05             # KL 目标

        # GRPO 超参数
        self.grpo_clip_eps = 0.2
        self.grpo_beta = 0.005
        self.grpo_entropy_weight = 0.001

    def policy_update(self, sample_data, epochs=1):
        """GRPO 策略更新"""
        state_batch, ref_probs_batch, advantages_batch, actions_batch, masks_batch = sample_data
        acc, kl, entropy = self.policy_value_net.train_step_grpo(
            state_batch, ref_probs_batch, advantages_batch, actions_batch, masks_batch,
            self.learn_rate * self.lr_multiplier,
            clip_eps=self.grpo_clip_eps,
            beta=self.grpo_beta,
            entropy_weight=self.grpo_entropy_weight
        )
        return acc, kl, entropy

    def run(self):
        """启动 GRPO 训练"""
        try:
            print("start data loader")
            self.dataset = GRPODataset(data_dir, self.buffer_size, -1, epochs=self.epochs)
            # test dataset shares data dict but uses newsample for validation
            self.testdataset = GRPOTestDataset(self.dataset)
            print("end data loader")

            try:
                self.policy_value_net = PolicyValueNet(
                    GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file, l2_const=1e-4
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                time.sleep(60)
                return

            self.policy_value_net.save_model(model_file + ".bak")

            dataset_len = len(self.dataset)
            training_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )
            testing_loader = torch.utils.data.DataLoader(
                self.testdataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

            # 训练前评估
            begin_accuracy = None
            begin_act_probs = None
            net = self.policy_value_net.policy_value
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, test_advs, test_action, test_mask = data
                if i == 0:
                    print("test_batch shape:", test_batch.shape, "test_probs shape:", test_probs.shape)
                test_batch = test_batch.to(self.policy_value_net.device)
                with torch.no_grad():
                    act_probs, _ = net(test_batch)
                    if begin_act_probs is None:
                        begin_act_probs = act_probs
                        begin_accuracy = np.argmax(act_probs, axis=1) == np.argmax(test_probs.cpu().numpy(), axis=1)
                    else:
                        begin_act_probs = np.concatenate((begin_act_probs, act_probs), axis=0)
                        begin_accuracy = np.concatenate(
                            (begin_accuracy, np.argmax(act_probs, axis=1) == np.argmax(test_probs.cpu().numpy(), axis=1)),
                            axis=0
                        )

            status = read_status_file()
            self.lr_multiplier = status["training"]["lr_multiplier"]
            print(f"lr_multiplier: {self.lr_multiplier}, learn_rate: {self.learn_rate * self.lr_multiplier}")

            # 训练循环
            for i, data in enumerate(training_loader):
                acc, kl, entropy = self.policy_update(data, self.epochs)
                if i % 10 == 0:
                    print(i, "acc:", acc, "kl:", kl, "entropy:", entropy)

                if i == 0:
                    state_batch, ref_probs_batch, advantages_batch, actions_batch, masks_batch = data
                    print("advantages_batch:", advantages_batch[:10])
                    print("actions_batch:", actions_batch[:10])

                if math.isnan(kl) or math.isnan(acc) or math.isnan(entropy) or \
                   math.isinf(kl) or math.isinf(acc) or math.isinf(entropy):
                    print(f"find nan or inf at step {i}!")
                    self.policy_value_net.save_model(model_file)
                    return

            self.policy_value_net.save_model(model_file)

            # 训练后评估
            end_accuracy = None
            end_act_probs = None
            all_test_probs = None
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, test_advs, test_action, test_mask = data
                test_batch = test_batch.to(self.policy_value_net.device)
                with torch.no_grad():
                    act_probs, _ = net(test_batch)
                    if all_test_probs is None:
                        all_test_probs = test_probs.cpu()
                    else:
                        all_test_probs = torch.cat((all_test_probs, test_probs.cpu()), dim=0)
                    if end_act_probs is None:
                        end_act_probs = act_probs
                        end_accuracy = np.argmax(act_probs, axis=1) == np.argmax(test_probs.cpu().numpy(), axis=1)
                    else:
                        end_act_probs = np.concatenate((end_act_probs, act_probs), axis=0)
                        end_accuracy = np.concatenate(
                            (end_accuracy, np.argmax(act_probs, axis=1) == np.argmax(test_probs.cpu().numpy(), axis=1)),
                            axis=0
                        )

            # 打印对比结果
            begin_act_probs_e = np.exp(begin_act_probs - np.max(begin_act_probs, axis=1, keepdims=True))
            begin_act_probs = begin_act_probs_e / np.sum(begin_act_probs_e, axis=1, keepdims=True)
            end_act_probs_e = np.exp(end_act_probs - np.max(end_act_probs, axis=1, keepdims=True))
            end_act_probs = end_act_probs_e / np.sum(end_act_probs_e, axis=1, keepdims=True)

            print(f"probs begin_accuracy: {np.mean(begin_accuracy):.4f} end_accuracy: {np.mean(end_accuracy):.4f}")

            # KL 散度：使用训练中最后一个 batch 的真实 KL
            status = read_status_file()
            set_status_value(status, "kl", kl, 0.1)
            total_kl = status["training"]["kl"]

            if total_kl > self.kl_targ * 2:
                self.lr_multiplier /= 1.1
            elif total_kl < self.kl_targ / 2:
                self.lr_multiplier *= 1.1
            self.lr_multiplier = np.clip(self.lr_multiplier, 0.1, 10)

            status["training"]["lr_multiplier"] = float(self.lr_multiplier)
            save_status_file(status)

            # ── test_play + EMA 指标更新 ──────────────────────────────
            print("running test_play for EMA metrics update...")
            sp = GRPOSelfPlay()
            sp.policy_value_net = self.policy_value_net
            (test_min_rl, _, _,
             test_max_rl, test_max_pc, test_min_pc,
             test_best_rl, test_worst_rl,
             test_avg_pc, test_avg_rl, test_avg_st) = sp.test_play()

            status = read_status_file()

            # 历史最值
            status["metrics"]["grpo_removedlines_best"] = max(status["metrics"]["grpo_removedlines_best"], test_best_rl)
            status["metrics"]["grpo_piececount_best"]   = max(status["metrics"]["grpo_piececount_best"],   test_max_pc)
            status["metrics"]["grpo_removedlines_worst"] = min(status["metrics"]["grpo_removedlines_worst"], test_worst_rl)
            status["metrics"]["grpo_piececount_worst"]   = min(status["metrics"]["grpo_piececount_worst"],   test_min_pc)

            # EMA 移动平均（贪婪策略）
            alpha = 0.1
            m = status["metrics"]
            m["grpo_piececount"]     = round(m.get("grpo_piececount",     0) * (1 - alpha) + test_avg_pc * alpha, 3)
            m["grpo_removedlines"]   = round(m.get("grpo_removedlines",   0) * (1 - alpha) + test_avg_rl * alpha, 3)
            m["grpo_steps"]          = round(m.get("grpo_steps",          0) * (1 - alpha) + test_avg_st * alpha, 3)

            # 最近一轮采集的奖励统计（从 dataset 的 advantage 计算）
            all_advs = np.array([step[2] for file_steps in self.dataset.data.values() for step in file_steps])
            m["grpo_reward_mean"] = round(float(all_advs.mean()), 3) if len(all_advs) > 0 else 0.0
            m["grpo_reward_std"]  = round(float(all_advs.std()),  3) if len(all_advs) > 0 else 0.0

            save_status_file(status)
            print(f"test: avg_pieces={test_avg_pc:.1f} avg_lines={test_avg_rl:.3f} avg_steps={test_avg_st:.1f}")
            print(f"EMA updated: pieces={m['grpo_piececount']} lines={m['grpo_removedlines']} steps={m['grpo_steps']}")

            print(f"kl:{kl:.6f} vs {self.kl_targ} lr_multiplier:{self.lr_multiplier} "
                  f"lr:{self.learn_rate * self.lr_multiplier}")

        except KeyboardInterrupt:
            print('quit')


if __name__ == '__main__':
    print('start GRPO training', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    training = GRPOTrain()
    training.run()
    print('end GRPO training', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
