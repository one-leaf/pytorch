import os, glob, pickle, shutil

from model import PolicyNet, data_dir, data_wait_dir, model_file, model_dir, log_nan
from agent import ACTIONS

import time
from datetime import datetime
import os, math, copy

import numpy as np
import torch

from status import read_train_state, save_train_state, set_train_value, train_file

# 定义游戏的动作
GAME_ACTIONS_NUM = len(ACTIONS)
GAME_WIDTH, GAME_HEIGHT = 10, 20


class PPODataset(torch.utils.data.Dataset):
    """PPO 数据集，每个 pkl 包含一局游戏的所有 step:
    (state, ref_prob, log_prob, action, prev_action, r_step, is_terminal, availables, v_t)
    load_data 后扩展为 9 元素: (state, ref_prob, log_prob, action, prev_action, r_step, is_terminal, availables, v_next)
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
        self.move_wait_files()
        self.load_game_files()
        self.load_data()

    def __len__(self):
        return len(self._flat_index)

    def __getitem__(self, index):
        fn, step_idx = self._flat_index[index]
        state, ref_prob, log_prob, action, prev_action, R, is_terminal, availables, v_next = self.data[fn][step_idx]
        return (torch.from_numpy(state).float(),
                torch.from_numpy(ref_prob).float(),
                torch.from_numpy(log_prob).float(),
                torch.as_tensor(action).long(),
                torch.as_tensor(prev_action).long(),
                torch.as_tensor(R).float(),
                torch.as_tensor(is_terminal).float(),
                torch.from_numpy(availables).float(),
                torch.as_tensor(v_next).float())

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
        """将所有 pkl 加载到内存，构建 flat index，并预计算每步的 v_next"""
        start_time = time.time()

        # ── 加载数据，预计算 v_next（下一步的 v_t）──
        for c, fn in enumerate(self.file_list):
            try:
                with open(fn, "rb") as f:
                    steps = pickle.load(f)

                n_steps = len(steps)
                if n_steps == 0:
                    print(f"file {fn} is empty, skipping")
                    continue

                # 检查数据格式：新格式 9 字段（包含 v_t）
                assert len(steps[0]) == 9, f'error: expected 9 elements, got {len(steps[0])} (old format, delete file)'

                # 预计算 v_next：下一步的 v_t，最后一步为 0
                v_nexts = np.zeros(n_steps)
                for t in range(n_steps - 1):
                    v_nexts[t] = steps[t + 1][8]  # 下一步的 v_t
                # v_nexts[-1] 保持为 0（游戏结束）

                # 扩展为 9 元素: (state, ref_prob, log_prob, action, prev_action, R, is_terminal, availables, v_next)
                steps_out = []
                for i, step in enumerate(steps):
                    steps_out.append((*step[:8], v_nexts[i]))

                self.data[fn] = steps_out
                if c == 0:
                    ref_probs = [step[1] for step in steps]
                    print(f"\n=== First file debug: {fn} ===")
                    print(f"R:         {[steps[i][5] for i in range(n_steps)]}")
                    print(f"V_next:    {list(v_nexts)}")
                    print(f"ref_probs: {[[round(max(rp), 3) for rp in ref_probs]]}")
                    print("=" * 40)
                                    
            except Exception as e:
                print(f"file {fn} error: {e}")
                if os.path.exists(fn):
                    os.remove(fn)
                self.file_list.remove(fn)

        # 统计 v_next 的分布（仅用于日志，不归一化）
        v_nexts = np.array([step[8] for steps in self.data.values() for step in steps])
        if len(v_nexts) > 0:
            print(f"v_next raw: mean={v_nexts.mean():.3f} std={v_nexts.std():.3f} min={v_nexts.min():.3f} max={v_nexts.max():.3f}")

        self._flat_index = [(fn, i) for fn in self.file_list for i in range(len(self.data[fn]))]

        print(f"loaded {len(self._flat_index)} steps in {time.time() - start_time:.1f}s")



class PPOTrain():
    def __init__(self):
        self.batch_size = 512
        self.learn_rate = 1e-5
        self.lr_multiplier = 1.0
        self.max_files = 10000          # data 目录最大保留文件数（安全上限，需 ≥ 2 × P × n_train_times）
        self.n_train_times = 2          # 每局严格保证被训练的轮数
        self.min_new_files = 1          # 至少有1个新文件就训练（不限制移动数量，清空 wait 目录）
        self.kl_targ = 0.02             # KL 超过 0.04 降速，低于 0.01 加速

        # PPO 超参数
        self.ppo_clip_eps = 0.2
        self.ppo_beta = 0.05            # KL 惩罚系数，beta*KL=0.05*0.4=0.02，与 policy_loss 可比
        self.ppo_entropy_weight = 1     # 熵正则（初始值，会自适应调整）
        self.entropy_target = 1.0       # 目标 entropy（自适应控制目标）
        self.entropy_ema = 1.0          # entropy EMA（用于自适应控制）
        self.n_epochs = 1               # 每轮训练只跑 1 个 epoch，训练次数由 min_new_files 控制

    def policy_update(self, sample_data):
        """PPO 策略更新"""
        state_batch, _ref_probs_batch, log_probs_old_batch, actions_batch, prev_actions_batch, R_batch, is_terminal_batch, availables_batch, v_next_batch = sample_data
        acc, kl, entropy, value_loss = self.policy_net.train_step_ppo(
            state_batch, log_probs_old_batch, actions_batch, prev_actions_batch,
            R_batch, is_terminal_batch, v_next_batch,
            self.learn_rate * self.lr_multiplier,
            clip_eps=self.ppo_clip_eps,
            beta=self.ppo_beta,
            entropy_weight=self.ppo_entropy_weight,
            availables_batch=availables_batch
        )
        return acc, kl, entropy, value_loss

    def run(self):
        """启动 PPO 训练"""
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
                    self.dataset = PPODataset(data_dir, self.max_files, self.min_new_files, self.n_train_times)
                    print("end data loader")
                    break
                except Exception as e:
                    print(f"waiting for data: {e}")
                    time.sleep(30)

            train_state = read_train_state()
            self.lr_multiplier = train_state["training"]["lr_multiplier"]
            self.ppo_entropy_weight = float(train_state["training"].get("entropy_weight", 1.0))
            self.entropy_ema = float(train_state["training"].get("entropy_ema", 1.0))
            print(f"batch_size: {self.batch_size}, lr_multiplier: {self.lr_multiplier}, entropy_weight: {self.ppo_entropy_weight}, entropy_ema: {self.entropy_ema}, learn_rate: {self.learn_rate * self.lr_multiplier}")

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
                        print(f"Train {i} {self.batch_size*i/len(self.dataset)*100:.1f}%",
                              f"acc:{acc:.4f} kl:{kl:.5f} ent:{entropy:.4f} vloss:{value_loss:.4f}",
                              f"ent_w:{self.ppo_entropy_weight:.3f} ent_ema:{self.entropy_ema:.4f}")

                    # if epoch == 0 and i == 0:
                    #     state_batch, ref_probs_batch, log_probs_old_batch, actions_batch, prev_actions_batch, R_batch, _is_terminal, _availables, v_next_batch = data
                    #     print("R_batch:", R_batch)
                    #     print("v_next_batch:", v_next_batch)
                    #     print("actions_batch:", actions_batch)
                    #     print("terminal:", _is_terminal)

                    if math.isnan(kl) or math.isnan(acc) or math.isnan(entropy) or math.isnan(value_loss) or \
                       math.isinf(kl) or math.isinf(acc) or math.isinf(entropy) or math.isinf(value_loss):
                        msg = f"LOSS NaN/Inf | epoch {epoch+1} step {i}: acc={acc} kl={kl} entropy={entropy} vloss={value_loss}"
                        print(f"\n[ROLLBACK] {msg}")
                        log_nan(msg)
                        # 从最佳模型目录还原（找方块数最大的子目录）
                        best_dirs = [d for d in os.listdir(model_dir)
                                     if os.path.isdir(os.path.join(model_dir, d)) and d.replace('.', '').isdigit()]
                        if best_dirs:
                            best_dirs.sort(key=lambda x: float(x), reverse=True)
                            best_dir = os.path.join(model_dir, best_dirs[0])
                            restore_model = os.path.join(best_dir, 'model.pth')
                            restore_train = os.path.join(best_dir, 'train.json')
                            if os.path.exists(restore_model):
                                print(f"[ROLLBACK] restoring from best dir: {best_dir}")
                                shutil.copy2(restore_model, model_file)
                                if os.path.exists(restore_train):
                                    shutil.copy2(restore_train, train_file)
                                self.policy_net = PolicyNet(
                                    GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=model_file, l2_const=1e-4
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

            # 训练结束后才更新 entropy_weight，供下一轮 train 使用
            # 目标熵：target_entropy = -scale × ln(|A|)，离散空间类比
            # 5 动作，scale=0.5~0.7 → target ≈ 0.8~1.13
            # 经验：0.8~1.0 折中；<0.5 策略定型；>1.3 基本随机
            entropy_diff = self.entropy_target - float(avg_ent)
            # 更新 entropy EMA
            self.entropy_ema = self.entropy_ema * 0.9 + avg_ent * 0.1
            if entropy_diff > 0.1:  # 只在低于目标熵 > 0.1 时调整
                adjust = 1.0 + 0.1 * entropy_diff  # 比例控制
                self.ppo_entropy_weight = float(np.clip(self.ppo_entropy_weight * adjust, 0.01, 1.0))
            elif entropy_diff < -0.2:  # 高于目标熵 > 0.2 时调整
                adjust = 1.0 + 0.01 * entropy_diff
                self.ppo_entropy_weight = float(np.clip(self.ppo_entropy_weight * adjust, 0.6, 1.0))
                
            print(f"entropy update: avg_ent={avg_ent:.4f} ema={self.entropy_ema:.4f} "
                  f"diff={entropy_diff:.4f} ent_w={self.ppo_entropy_weight:.3f}")
            avg_vl  = _sum_vl  / max(_num_batches, 1)

            self.policy_net.save_model(model_file)

            # KL 散度：使用训练循环的平均 KL
            train_state = read_train_state()
            alpha = 0.1
            m = train_state["metrics"]
            m["train_acc"]     = round(m.get("train_acc",     0) * (1 - alpha) + avg_acc * alpha, 5)
            m["train_kl"]      = round(m.get("train_kl",      0) * (1 - alpha) + avg_kl  * alpha, 5)
            m["train_entropy"] = round(m.get("train_entropy", 0) * (1 - alpha) + avg_ent * alpha, 5)
            m["train_vloss"]   = round(m.get("train_vloss",   0) * (1 - alpha) + avg_vl  * alpha, 5)
            # lr_multiplier 调整使用 EMA 平滑后的 train_kl
            set_train_value(train_state, "kl", avg_kl, alpha)
            total_kl = train_state["training"]["kl"]

            if total_kl > self.kl_targ * 2:
                self.lr_multiplier /= 1.1
            elif total_kl < self.kl_targ / 2:
                self.lr_multiplier *= 1.1
            self.lr_multiplier = np.clip(self.lr_multiplier, 0.1, 3.0)

            train_state["training"]["lr_multiplier"] = float(self.lr_multiplier)
            train_state["training"]["entropy_weight"] = float(self.ppo_entropy_weight)
            train_state["training"]["entropy_ema"] = float(self.entropy_ema)
            train_state["counters"]["train"] += 1
            train_state["counters"]["_train"] += 1
            save_train_state(train_state)
            print(f"train EMA: acc={m['train_acc']:.4f} kl={m['train_kl']:.5f} entropy={m['train_entropy']:.4f} vloss={m['train_vloss']:.4f}")

            print(f"kl:{kl:.6f} vs {self.kl_targ} lr_multiplier:{self.lr_multiplier} "
                  f"lr:{self.learn_rate * self.lr_multiplier}")

        except KeyboardInterrupt:
            print('quit')


if __name__ == '__main__':
    print('start PPO training', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    training = PPOTrain()
    training.run()
    print('end PPO training', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
