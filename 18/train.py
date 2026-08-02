import os, glob, pickle, shutil

from model import PolicyNet, data_dir, data_wait_dir, model_file, model_dir, log_nan
from agent import ACTIONS

import time
from datetime import datetime
import os, math, copy

import numpy as np
import torch

from status import save_status_file, read_status_file, set_status_value, status_file

# 定义游戏的动作
GAME_ACTIONS_NUM = len(ACTIONS)
GAME_WIDTH, GAME_HEIGHT = 10, 20


class PPODataset(torch.utils.data.Dataset):
    """PPO 数据集，每个 pkl 包含一局游戏的所有 step:
    (state, ref_prob, log_prob, action, prev_action, R, is_terminal, availables)
    game_id 由文件名推导，不在 pkl 中存储
    load_data 后扩展为 9 元素: (state, ref_prob, log_prob, action, prev_action, R, is_terminal, availables, G)
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
        state, ref_prob, log_prob, action, prev_action, R, is_terminal, availables, G = self.data[fn][step_idx]
        game_id = os.path.basename(fn)  # 用文件名作为 game_id
        return (torch.from_numpy(state).float(),
                torch.from_numpy(ref_prob).float(),
                torch.from_numpy(log_prob).float(),
                torch.as_tensor(action).long(),
                torch.as_tensor(prev_action).long(),
                game_id,
                torch.as_tensor(R).float(),
                torch.as_tensor(is_terminal).float(),
                torch.from_numpy(availables).float(),
                torch.as_tensor(G).float())

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
        """将所有 pkl 加载到内存，构建 flat index，并预计算每步的 G_t"""
        start_time = time.time()
        gamma = 0.99

        # ── Pass 1: 收集每局最后一步的 r（方块数），用于归一化 ──
        game_piececounts = {}  # fn → piececount（最后一步的 r）
        for fn in self.file_list:
            try:
                with open(fn, "rb") as f:
                    steps = pickle.load(f)
                # (state, ref_prob, log_prob, action, prev_action, r_step, is_terminal, availables)                    
                assert len(steps[0]) == 8, f'error: expected 8 elements, got {len(steps[0])} (old format, delete file)'
                game_piececounts[fn] = steps[-1][5]  # 最后一步的 r = 方块数
            except Exception as e:
                print(f"file {fn} scan error: {e}")
                if os.path.exists(fn):
                    os.remove(fn)
                self.file_list.remove(fn)

        if not game_piececounts:
            return

        all_pc = list(game_piececounts.values())
        min_pc = min(all_pc)
        max_pc = max(all_pc)
        pc_range = max(max_pc - min_pc, 1)
        print(f"Piececounts: min={min_pc} max={max_pc} mean={np.mean(all_pc):.1f} games={len(all_pc)}")

        # ── Pass 2: 加载数据，归一化终止奖励到 [-1, 1]，计算 G_t ──
        for fn in self.file_list:
            if fn not in game_piececounts:
                continue
            try:
                with open(fn, "rb") as f:
                    steps = pickle.load(f)

                n_steps = len(steps)

                # 归一化终止奖励：方块数越多 → 越接近 +1，方块数越少 → 越接近 -1
                # 落地消行奖励（中间步骤）保持不变
                pc = game_piececounts[fn]
                terminal_r = 2.0 * (pc - min_pc) / pc_range - 1.0

                # 修改最后一步的 R
                last_step = steps[-1]
                steps[-1] = (*last_step[:5], terminal_r, last_step[6], last_step[7])

                # 预计算这局游戏的 G_t（折扣回报）
                g_values = np.zeros(n_steps)
                g_values[-1] = steps[-1][5]
                for t in range(n_steps - 2, -1, -1):
                    g_values[t] = steps[t][5] + gamma * g_values[t + 1]

                # 扩展为 9 元素: (state, ref_prob, log_prob, action, prev_action, R, is_terminal, availables, G)
                steps_out = []
                for i, step in enumerate(steps):
                    steps_out.append((*step[:8], g_values[i]))

                self.data[fn] = steps_out
            except Exception as e:
                print(f"file {fn} error: {e}")
                if os.path.exists(fn):
                    os.remove(fn)
                self.file_list.remove(fn)

        Rs = np.array([step[5] for steps in self.data.values() for step in steps])
        Gs = np.array([step[8] for steps in self.data.values() for step in steps])
        if len(Rs) > 0:
            # 保存原始统计用于跟踪
            self.g_mean_raw = float(Gs.mean())
            self.g_std_raw = max(float(Gs.std()), 1e-3)
            print(f"R raw: min={Rs.min():.1f} max={Rs.max():.1f}")
            print(f"G raw: min={Gs.min():.1f} mean={self.g_mean_raw:.2f} std={self.g_std_raw:.2f} max={Gs.max():.1f}")

            # 直接写入 status（不平滑）
            status = read_status_file()
            m = status["metrics"]
            m["g_mean_raw"] = round(self.g_mean_raw, 3)
            m["g_std_raw"]  = round(self.g_std_raw, 3)
            save_status_file(status)

            # 归一化 G（数据集级别 z-score）
            g_mean, g_std = self.g_mean_raw, self.g_std_raw
            for fn_key in self.data:
                self.data[fn_key] = [
                    (s[0], s[1], s[2], s[3], s[4],
                     s[5],
                     s[6], s[7],
                     (s[8] - g_mean) / g_std)
                    for s in self.data[fn_key]
                ]
            print(f"G normalized: g→(0,1)")
        else:
            self.g_mean_raw = 0.0
            self.g_std_raw = 1.0

        self._flat_index = [(fn, i) for fn in self.file_list for i in range(len(self.data[fn]))]

        print(f"loaded {len(self._flat_index)} steps in {time.time() - start_time:.1f}s")



class PPOTrain():
    def __init__(self):
        self.batch_size = 512
        self.learn_rate = 1e-5
        self.lr_multiplier = 1.0
        self.max_files = 20000          # data 目录最大保留文件数（安全上限，需 ≥ 2 × P × n_train_times）
        self.n_train_times = 2          # 每局严格保证被训练的轮数
        self.min_new_files = 1          # 至少有1个新文件就训练（不限制移动数量，清空 wait 目录）
        self.kl_targ = 0.02             # KL 超过 0.04 降速，低于 0.01 加速

        # PPO 超参数
        self.ppo_clip_eps = 0.2
        self.ppo_beta = 0.05            # KL 惩罚系数，beta*KL=0.05*0.4=0.02，与 policy_loss 可比
        self.ppo_entropy_weight = 1.0   # 熵正则（初始值，会自适应调整）
        self.entropy_target = 1.0       # 目标 entropy（自适应控制目标）
        self.entropy_ema = 1.0          # entropy EMA（用于自适应控制）
        self.n_epochs = 1               # 每轮训练只跑 1 个 epoch，训练次数由 min_new_files 控制

    def policy_update(self, sample_data):
        """PPO 策略更新（带 GAE 信用分配）"""
        state_batch, _ref_probs_batch, log_probs_old_batch, actions_batch, prev_actions_batch, game_ids_batch, R_batch, is_terminal_batch, availables_batch, G_batch = sample_data
        acc, kl, entropy, value_loss = self.policy_net.train_step_ppo(
            state_batch, log_probs_old_batch, actions_batch, prev_actions_batch,
            game_ids_batch, R_batch, is_terminal_batch, G_batch,
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

            status = read_status_file()
            self.lr_multiplier = status["training"]["lr_multiplier"]
            self.ppo_entropy_weight = float(status["training"].get("entropy_weight", 1.0))
            print(f"batch_size: {self.batch_size}, lr_multiplier: {self.lr_multiplier}, entropy_weight: {self.ppo_entropy_weight}, learn_rate: {self.learn_rate * self.lr_multiplier}")

            # 训练循环（n_epochs 个 epoch，保证每局被训练 n_epochs 次）
            _sum_acc = _sum_kl = _sum_ent = _sum_vl = 0.0
            _num_batches = 0
            for epoch in range(self.n_epochs):
                # ⚠️ 必须 shuffle=False: GAE 要求同一 game 的步骤按时间顺序排列
                training_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
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

                    # 自适应 entropy_weight（SAC 启发式）
                    # 目标熵：target_entropy = -scale × ln(|A|)，离散空间类比
                    # 5 动作，scale=0.5~0.7 → target ≈ 0.8~1.13
                    # 经验：0.8~1.0 折中；<0.5 策略定型；>1.3 基本随机
                    self.entropy_ema = 0.99 * self.entropy_ema + 0.01 * float(entropy)
                    entropy_diff = self.entropy_target - self.entropy_ema
                    if abs(entropy_diff) > 0.05:  # 只在偏差 > 0.05 时调整
                        adjust = 1.0 + 0.1 * entropy_diff  # 比例控制
                        self.ppo_entropy_weight = float(np.clip(self.ppo_entropy_weight * adjust, 0.1, 5.0))

                    if i % 500 == 0:
                        print(f"Train {i} {self.batch_size*i/len(self.dataset)*100:.1f}%",
                              f"acc:{acc:.4f} kl:{kl:.5f} ent:{entropy:.4f} vloss:{value_loss:.4f}",
                              f"ent_w:{self.ppo_entropy_weight:.3f} ent_ema:{self.entropy_ema:.4f}")

                    if epoch == 0 and i == 0:
                        state_batch, ref_probs_batch, log_probs_old_batch, actions_batch, prev_actions_batch, game_ids_batch, R_batch, _is_terminal, _availables, G_batch = data
                        print("R_batch:", R_batch)
                        print("G_batch:", G_batch)
                        print("actions_batch:", actions_batch)
                        print("terminal:", _is_terminal)
                        print("game_ids_batch:", set(game_ids_batch))

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
                            restore_status = os.path.join(best_dir, 'status.json')
                            if os.path.exists(restore_model):
                                print(f"[ROLLBACK] restoring from best dir: {best_dir}")
                                shutil.copy2(restore_model, model_file)
                                if os.path.exists(restore_status):
                                    shutil.copy2(restore_status, status_file)
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
            avg_vl  = _sum_vl  / max(_num_batches, 1)

            self.policy_net.save_model(model_file)

            # KL 散度：使用训练循环的平均 KL
            status = read_status_file()
            alpha = 0.1
            m = status["metrics"]
            m["train_acc"]     = round(m.get("train_acc",     0) * (1 - alpha) + avg_acc * alpha, 5)
            m["train_kl"]      = round(m.get("train_kl",      0) * (1 - alpha) + avg_kl  * alpha, 5)
            m["train_entropy"] = round(m.get("train_entropy", 0) * (1 - alpha) + avg_ent * alpha, 5)
            m["train_vloss"]   = round(m.get("train_vloss",   0) * (1 - alpha) + avg_vl  * alpha, 5)
            # lr_multiplier 调整使用 EMA 平滑后的 train_kl
            set_status_value(status, "kl", avg_kl, alpha)
            total_kl = status["training"]["kl"]

            if total_kl > self.kl_targ * 2:
                self.lr_multiplier /= 1.1
            elif total_kl < self.kl_targ / 2:
                self.lr_multiplier *= 1.1
            self.lr_multiplier = np.clip(self.lr_multiplier, 0.1, 3.0)

            status["training"]["lr_multiplier"] = float(self.lr_multiplier)
            status["training"]["entropy_weight"] = float(self.ppo_entropy_weight)
            status["counters"]["train"] += 1
            status["counters"]["_train"] += 1
            save_status_file(status)
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
