import os, glob, pickle

from model import PolicyNet, data_dir, data_wait_dir, model_file, log_nan
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
    """GRPO 数据集，每个 pkl 包含一局游戏的所有 step:
    (state, ref_prob, log_prob, action, prev_action, game_id, R)
    """
    def __init__(self, data_dir, max_files, min_new_files, n_train_times=3):
        self.data_dir = data_dir
        self.max_files = max_files
        self.min_new_files = min_new_files
        self.n_train_times = n_train_times
        self.file_list = []
        self.newsample = []
        self.data = {}
        self._flat_index = []
        self._test_flat_index = []
        self.move_wait_files()
        self.load_game_files()
        self.load_data()

    def __len__(self):
        return len(self._flat_index)

    def __getitem__(self, index):
        fn, step_idx = self._flat_index[index]
        state, ref_prob, log_prob, action, prev_action, game_id, R = self.data[fn][step_idx]
        return (torch.from_numpy(state).float(),
                torch.from_numpy(ref_prob).float(),
                torch.from_numpy(log_prob).float(),
                torch.as_tensor(action).long(),
                torch.as_tensor(prev_action).long(),
                torch.as_tensor(game_id).long(),
                torch.as_tensor(R).float())

    def move_wait_files(self):
        """将 wait 目录的 pkl 全部移入 data 目录（清空 wait，防止堆积）"""
        files = sorted(glob.glob(os.path.join(data_wait_dir, "*.pkl")),
                       key=lambda x: os.path.getmtime(x))
        time.sleep(1)

        if len(files) < self.min_new_files:
            print(f"Insufficient data: have {len(files)}, need {self.min_new_files}")
            raise Exception("NEED MORE DATA TO TRAIN")

        for fn in files:
            dest = os.path.join(self.data_dir, os.path.basename(fn))
            if os.path.exists(dest):
                os.remove(dest)
            os.rename(fn, dest)
            self.newsample.append(dest)

        print(f"moved {len(files)} files to train, newsample: {len(self.newsample)}")

    def load_game_files(self):
        """加载 data 目录的文件列表，按时间倒序，动态删除以保证每局被训练 n_train_times 次"""
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.pkl")),
                       key=lambda x: os.path.getmtime(x), reverse=True)

        if not files:
            print("no data files found")
            return

        print(f"first time: {time.strftime('%y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(files[-1])))}")
        print(f"last time:  {time.strftime('%y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(files[0])))}")

        # 动态删除：每轮删除 newsample // n_train_times 个最旧文件
        # 数据池稳定在 (T-1)*P，每局存活恰好 T 轮
        n_new = len(self.newsample)
        to_delete_by_rotation = n_new // self.n_train_times if n_new >= self.n_train_times else 0
        keep_count = min(len(files), self.max_files)
        keep_count = max(0, keep_count - to_delete_by_rotation)

        for i, filename in enumerate(files):
            if i < keep_count and os.path.getsize(filename) > 0:
                self.file_list.append(filename)
            else:
                os.remove(filename)

        deleted = len(files) - len(self.file_list)
        print(f"loaded {len(self.file_list)} files, deleted {deleted} (pool rotation: {to_delete_by_rotation})")

    def load_data(self):
        """将所有 pkl 加载到内存，构建 flat index"""
        start_time = time.time()
        for fn in self.file_list:
            try:
                with open(fn, "rb") as f:
                    steps = pickle.load(f)
                for step in steps:
                    state, ref_prob, _log_prob, _action, _prev_action, _game_id, R = step
                    assert state.shape == (2, 20, 10), f'error: state shape {state.shape}'
                    assert ref_prob.shape == (5,), f'error: ref_prob shape {ref_prob.shape}'
                    assert not np.isnan(R), f'error: R is Nan'
                self.data[fn] = steps
            except Exception as e:
                print(f"file {fn} error: {e}")
                if os.path.exists(fn):
                    os.remove(fn)
                self.file_list.remove(fn)

        Rs = np.array([step[6] for steps in self.data.values() for step in steps])
        if len(Rs) > 0:
            self.r_mean = float(Rs.mean())
            self.r_std = max(float(Rs.std()), 1e-3)
            print(f"R stats: min={Rs.min():.1f} mean={self.r_mean:.2f} std={self.r_std:.2f} max={Rs.max():.1f}")
        else:
            self.r_mean = 15.0
            self.r_std = 1.0

        self._flat_index = [(fn, i) for fn in self.file_list for i in range(len(self.data[fn]))]
        self._test_flat_index = [(fn, i) for fn in self.newsample for i in range(len(self.data.get(fn, [])))]

        print(f"loaded {len(self._flat_index)} steps in {time.time() - start_time:.1f}s")


class GRPOTestDataset(torch.utils.data.Dataset):
    """测试数据集：使用 newsample（本轮新采集的文件）"""
    def __init__(self, parent: GRPODataset):
        self.parent = parent

    def __len__(self):
        return len(self.parent._test_flat_index)

    def __getitem__(self, index):
        fn, step_idx = self.parent._test_flat_index[index]
        state, ref_prob, log_prob, action, prev_action, game_id, R = self.parent.data[fn][step_idx]
        return (torch.from_numpy(state).float(),
                torch.from_numpy(ref_prob).float(),
                torch.from_numpy(log_prob).float(),
                torch.as_tensor(action).long(),
                torch.as_tensor(prev_action).long(),
                torch.as_tensor(game_id).long(),
                torch.as_tensor(R).float())


class GRPOTrain():
    def __init__(self):
        self.batch_size = 256
        self.learn_rate = 1e-5
        self.lr_multiplier = 1.0
        self.max_files = 15000         # data 目录最大保留文件数（安全上限，需 ≥ 2 × P × n_train_times）
        self.n_train_times = 3         # 每局严格保证被训练的轮数
        self.min_new_files = 1         # 至少有1个新文件就训练（不限制移动数量，清空 wait 目录）
        self.kl_targ = 0.3           # 实际 KL 约 0.6~0.8，目标设为 0.3 允许充分学习

        # GRPO 超参数
        self.grpo_clip_eps = 0.2
        self.grpo_beta = 0.02         # KL 惩罚系数，beta*KL=0.02*0.8=0.016，与 policy_loss 量级可比
        self.grpo_entropy_weight = 0.2   # 熵正则，维持 entropy 在 0.8-1.0 促进探索
        self.n_epochs = 1             # 每轮训练只跑 1 个 epoch，训练次数由 min_new_files 控制

    def policy_update(self, sample_data):
        """GRPO 策略更新（带 GAE 信用分配）"""
        state_batch, ref_probs_batch, log_probs_old_batch, actions_batch, prev_actions_batch, game_ids_batch, R_batch = sample_data
        acc, kl, entropy, value_loss = self.policy_net.train_step_grpo(
            state_batch, ref_probs_batch, log_probs_old_batch, actions_batch, None, prev_actions_batch,
            game_ids_batch, R_batch,
            self.learn_rate * self.lr_multiplier,
            self.dataset.r_mean, self.dataset.r_std,
            clip_eps=self.grpo_clip_eps,
            beta=self.grpo_beta,
            entropy_weight=self.grpo_entropy_weight
        )
        return acc, kl, entropy, value_loss

    def run(self):
        """启动 GRPO 训练"""
        try:
            # 先创建/加载模型（确保 model_file 存在，selfplay 才能启动）
            try:
                self.policy_net = PolicyNet(
                    GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file, l2_const=1e-4
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                time.sleep(60)
                return

            # 等待 selfplay 产生训练数据
            while True:
                try:
                    print("start data loader")
                    self.dataset = GRPODataset(data_dir, self.max_files, self.min_new_files, self.n_train_times)
                    self.testdataset = GRPOTestDataset(self.dataset)
                    print("end data loader")
                    break
                except Exception as e:
                    print(f"waiting for data: {e}")
                    time.sleep(30)

            dataset_len = len(self.dataset)
            testing_loader = torch.utils.data.DataLoader(
                self.testdataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )

            # 训练前评估
            begin_accuracy = np.array([])
            begin_act_probs = None
            net = self.policy_net.policy
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, _log_probs_old, test_action, test_prev_action, _game_ids, _R = data
                if i == 0:
                    print("test_batch shape:", test_batch.shape, "test_probs shape:", test_probs.shape)
                test_batch = test_batch.to(self.policy_net.device)
                test_prev_action = test_prev_action.to(self.policy_net.device)
                with torch.no_grad():
                    act_probs = net(test_batch, test_prev_action)
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
            print(f"batch_size: {self.batch_size}, lr_multiplier: {self.lr_multiplier}, learn_rate: {self.learn_rate * self.lr_multiplier}")

            # 训练循环（n_epochs 个 epoch，保证每局被训练 n_epochs 次）
            _sum_acc = _sum_kl = _sum_ent = _sum_vl = 0.0
            _num_batches = 0
            for epoch in range(self.n_epochs):
                training_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
                )
                _epoch_acc = _epoch_kl = _epoch_ent = _epoch_vl = 0.0
                _epoch_batches = 0
                for i, data in enumerate(training_loader):
                    acc, kl, entropy, value_loss = self.policy_update(data)
                    _sum_acc += acc
                    _sum_kl += kl
                    _sum_ent += entropy
                    _sum_vl += value_loss
                    _num_batches += 1
                    _epoch_acc += acc
                    _epoch_kl += kl
                    _epoch_ent += entropy
                    _epoch_vl += value_loss
                    _epoch_batches += 1
                    if i % 100 == 0:
                        print(f"epoch {epoch+1}/{self.n_epochs}", i,
                              "acc:", acc, "kl:", kl, "entropy:", entropy, "vloss:", value_loss,
                              "r_mean:", round(self.dataset.r_mean, 2), "r_std:", round(self.dataset.r_std, 2))

                    if epoch == 0 and i == 0:
                        state_batch, ref_probs_batch, log_probs_old_batch, actions_batch, prev_actions_batch, game_ids_batch, R_batch = data
                        print("R_batch:", R_batch[:10])
                        print("actions_batch:", actions_batch[:10])

                    if math.isnan(kl) or math.isnan(acc) or math.isnan(entropy) or math.isnan(value_loss) or \
                       math.isinf(kl) or math.isinf(acc) or math.isinf(entropy) or math.isinf(value_loss):
                        msg = f"LOSS NaN/Inf | epoch {epoch+1} step {i}: acc={acc} kl={kl} entropy={entropy} vloss={value_loss}"
                        print(f"\n[ROLLBACK] {msg}")
                        log_nan(msg)
                        if os.path.exists(model_file + ".bak"):
                            print(f"[ROLLBACK] restoring from {model_file}.bak")
                            self.policy_net = PolicyNet(
                                GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file + ".bak", l2_const=1e-4
                            )
                        return
                e_acc = _epoch_acc / max(_epoch_batches, 1)
                e_kl  = _epoch_kl  / max(_epoch_batches, 1)
                e_ent = _epoch_ent / max(_epoch_batches, 1)
                e_vl  = _epoch_vl  / max(_epoch_batches, 1)
                print(f"epoch {epoch+1} done: acc={e_acc:.4f} kl={e_kl:.5f} entropy={e_ent:.4f} vloss={e_vl:.4f}")

            avg_acc = _sum_acc / max(_num_batches, 1)
            avg_kl  = _sum_kl  / max(_num_batches, 1)
            avg_ent = _sum_ent / max(_num_batches, 1)
            avg_vl  = _sum_vl  / max(_num_batches, 1)
            avg_r_mean = self.dataset.r_mean
            avg_r_std = self.dataset.r_std

            self.policy_net.save_model(model_file)

            # 训练后评估
            end_accuracy = None
            end_act_probs = None
            all_test_probs = None
            end_accuracy = np.array([])
            for i, data in enumerate(testing_loader):
                test_batch, test_probs, _log_probs_old, _test_action, test_prev_action, _game_ids, _R = data
                test_batch = test_batch.to(self.policy_net.device)
                test_prev_action = test_prev_action.to(self.policy_net.device)
                with torch.no_grad():
                    act_probs = net(test_batch, test_prev_action)
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

            # KL 散度：使用训练循环的平均 KL
            status = read_status_file()
            alpha = 0.1
            m = status["metrics"]
            m["train_acc"]     = round(m.get("train_acc",     0) * (1 - alpha) + avg_acc * alpha, 5)
            m["train_kl"]      = round(m.get("train_kl",      0) * (1 - alpha) + avg_kl  * alpha, 5)
            m["train_entropy"] = round(m.get("train_entropy", 0) * (1 - alpha) + avg_ent * alpha, 5)
            m["train_vloss"]   = round(m.get("train_vloss",   0) * (1 - alpha) + avg_vl  * alpha, 5)
            m["r_mean"]        = round(avg_r_mean, 3)
            m["r_std"]         = round(avg_r_std, 3)
            # lr_multiplier 调整使用 EMA 平滑后的 train_kl
            set_status_value(status, "kl", avg_kl, alpha)
            total_kl = status["training"]["kl"]

            if total_kl > self.kl_targ * 2:
                self.lr_multiplier /= 1.1
            elif total_kl < self.kl_targ / 2:
                self.lr_multiplier *= 1.1
            self.lr_multiplier = np.clip(self.lr_multiplier, 0.5, 5.0)

            status["training"]["lr_multiplier"] = float(self.lr_multiplier)
            save_status_file(status)
            print(f"train EMA: acc={m['train_acc']:.4f} kl={m['train_kl']:.5f} entropy={m['train_entropy']:.4f} vloss={m['train_vloss']:.4f} r_mean={m['r_mean']:.2f} r_std={m['r_std']:.2f}")

            # ── test_play + EMA 指标更新 ──────────────────────────────
            print("running test_play for EMA metrics update...")
            sp = GRPOSelfPlay()
            sp.policy_net = self.policy_net
            (_, _, _,
             _, _, _,
             test_best_rl, _,
             test_avg_pc, test_avg_rl, test_avg_st) = sp.test_play()

            status = read_status_file()
            alpha = 0.1
            m = status["metrics"]

            # test_play EMA（纯贪婪，无噪声）
            old_ema_pc = m.get("test_piececount", 0)
            m["test_piececount"]       = round(old_ema_pc * (1 - alpha) + test_avg_pc * alpha, 3)
            m["test_removedlines"]     = round(m.get("test_removedlines",     0) * (1 - alpha) + test_avg_rl * alpha, 3)
            m["test_steps"]            = round(m.get("test_steps",            0) * (1 - alpha) + test_avg_st * alpha, 3)
            # test_play 历史最值（无噪声真实表现）
            m["test_piececount_best"]    = max(m.get("test_piececount_best",  0), test_avg_pc)
            m["test_removedlines_best"]  = max(m.get("test_removedlines_best",  0), test_best_rl)

            if test_avg_pc > old_ema_pc:
                best_model_path = f"{model_file}.{test_avg_pc:.1f}"
                self.policy_net.save_model(best_model_path)
                self.policy_net.save_model(model_file + ".bak")
                print(f"*** new best! test_avg_pc={test_avg_pc:.1f} > ema={old_ema_pc:.1f}, saved to {best_model_path}")

            # 最近一轮采集的奖励统计（从 dataset 的 R 值计算）
            all_rs = np.array([step[6] for file_steps in self.dataset.data.values() for step in file_steps])
            m["grpo_reward_mean"] = round(float(all_rs.mean()), 3) if len(all_rs) > 0 else 0.0
            m["grpo_reward_std"]  = round(float(all_rs.std()),  3) if len(all_rs) > 0 else 0.0

            save_status_file(status)
            print(f"test: greedy avg_pieces={test_avg_pc:.1f} avg_lines={test_avg_rl:.3f} avg_steps={test_avg_st:.1f}")
            print(f"test EMA: pieces={m['test_piececount']} lines={m['test_removedlines']} "
                  f"best_pieces={m['test_piececount_best']} best_lines={m['test_removedlines_best']}")

            print(f"kl:{kl:.6f} vs {self.kl_targ} lr_multiplier:{self.lr_multiplier} "
                  f"lr:{self.learn_rate * self.lr_multiplier}")

        except KeyboardInterrupt:
            print('quit')


if __name__ == '__main__':
    print('start GRPO training', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    training = GRPOTrain()
    training.run()
    print('end GRPO training', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
