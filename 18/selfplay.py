import os, pickle, time, itertools, shutil
from datetime import datetime
import numpy as np
import torch

from model import PolicyNet, data_wait_dir, model_file, model_dir
from agent import Agent, ACTIONS
from status import save_play_state, read_play_state, play_file, train_file

# 定义游戏的动作
GAME_ACTIONS_NUM = len(ACTIONS)
GAME_WIDTH, GAME_HEIGHT = 10, 20


class PPOSelfPlay():
    def __init__(self):
        self.rollout_max_steps = 500    # 单局最大步数
        self.policy_net = None

    def get_actions_batch(self, agents, prev_actions, greedy_indices=None):
        """批量预测多个游戏的动作（一次 forward pass）"""
        if greedy_indices is None:
            greedy_indices = set()
        # 只处理未结束的游戏
        active_indices = [i for i, a in enumerate(agents) if not a.terminal]
        if not active_indices:
            return []

        device = self.policy_net.device
        states = np.array([agents[i].current_state() for i in active_indices])
        prev_acts = [prev_actions[i] for i in active_indices]

        states_tensor = torch.FloatTensor(states).to(device)
        prev_tensor = torch.LongTensor(prev_acts).to(device)

        self.policy_net.net.eval()
        with torch.no_grad():
            log_probs_batch, _ = self.policy_net.net(states_tensor, prev_tensor)

        if torch.isnan(log_probs_batch).any():
            log_probs_batch = torch.zeros_like(log_probs_batch)

        actions = []
        all_probs = []
        all_log_probs = []
        all_availables = []

        for idx, i in enumerate(active_indices):
            agent = agents[i]
            log_probs = log_probs_batch[idx]
            availables = agent.availables

            if i not in greedy_indices:
                scaled = torch.exp(log_probs).cpu().numpy()
                scaled = np.clip(scaled, a_min=0.01, a_max=0.95)
                probs = scaled * availables.astype(np.float32)
                probs_sum = probs.sum()
                if probs_sum < 1e-10:
                    probs = availables.astype(np.float32)
                    probs_sum = probs.sum()
                probs = probs / probs_sum
                action = np.random.choice(GAME_ACTIONS_NUM, p=probs)
            else:
                probs = torch.exp(log_probs).cpu().numpy()
                probs = probs * availables.astype(np.float32)
                action = np.argmax(probs)

            actions.append(int(action))
            all_probs.append(probs.copy())
            # 记录模型原始 log_probs（不做 clamp，由 train_step_ppo 统一处理）
            all_log_probs.append(log_probs.cpu().numpy())
            all_availables.append(availables.astype(np.float32))

        return actions, all_probs, all_log_probs, all_availables

    def play_games_parallel(self, n_games=4, pieces_list=None, greedy_indices=None):
        """同时玩 n_games 局，共享方块序列，批量预测"""
        agents = [Agent(isRandomNextPiece=False, nextPiecesList=pieces_list or []) for _ in range(n_games)]
        trajectories = [[] for _ in range(n_games)]
        step_results = [[] for _ in range(n_games)]
        prev_actions = [3] * n_games  # KEY_NONE

        for _ in range(self.rollout_max_steps):
            if all(a.terminal for a in agents):
                break

            actions, all_probs, all_log_probs, all_availables = self.get_actions_batch(
                agents, prev_actions, greedy_indices
            )

            # 为每个 active 游戏记录轨迹
            action_idx = 0
            for i, agent in enumerate(agents):
                if agent.terminal:
                    continue

                state = agent.current_state().copy()
                action = actions[action_idx]
                probs = all_probs[action_idx]
                log_prob = all_log_probs[action_idx]
                availables = all_availables[action_idx]
                action_idx += 1

                trajectories[i].append({
                    "state": state,
                    "action": action,
                    "prev_action": prev_actions[i],
                    "ref_prob": probs,
                    "log_prob": log_prob,
                    "availables": availables,
                })

                prev_actions[i] = action
                landed, removed = agent.step(action)
                step_results[i].append((landed, removed))

        return agents, trajectories, step_results

    def _check_and_fix_nan(self, policy_net):
        """检测模型是否输出 NaN，如有则重新初始化权重"""
        device = policy_net.device
        dummy_state = torch.zeros(1, 2, 20, 10, device=device)
        dummy_prev = torch.zeros(1, dtype=torch.long, device=device)
        with torch.no_grad():
            out, _ = policy_net.net(dummy_state, dummy_prev)
        if torch.isnan(out).any():
            print("WARNING: model output contains NaN, reinitializing weights!")
            policy_net.net.init_weights()

    def collect_ppo_data(self):
        """收集 PPO 自我对抗数据"""
        print("PPO Self Play starting ...")

        # 确定初始模型文件
        load_model_file = model_file

        # 等待模型文件出现
        while not os.path.exists(load_model_file):
            print("no model file found, waiting for train to create one... (sleep 30s)")
            time.sleep(30)

        if time.time() - os.path.getmtime(load_model_file) > 60 * 60 * 5:
            print("超过5小时模型都没有更新了，停止训练")
            return

        # 加载模型用于数据收集
        if self.policy_net is None:
            self.policy_net = PolicyNet(
                GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=load_model_file
            )
        self._check_and_fix_nan(self.policy_net)
        _last_model_mtime = os.path.getmtime(load_model_file)

        # 持续采集，每局完成后立即保存
        print("starting continuous collection ...")
        _start_time = time.time()
        game_counter = 0

        for _ in itertools.count():
            if time.time() - _start_time > 60 * 60:  # 最多60分钟采集
                break

            # 每组之前检查模型是否有更新，有则重新加载
            current_model = model_file
            if os.path.exists(current_model):
                mtime = os.path.getmtime(current_model)
                if mtime > _last_model_mtime:
                    print(f"Model updated, reloading from {current_model}")
                    self.policy_net = PolicyNet(
                        GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=current_model
                    )
                    self._check_and_fix_nan(self.policy_net)
                    _last_model_mtime = mtime

            # 并行玩 16 局（game 0 贪婪测试，game 1-15 带 V(s) 温度探索）
            agents, trajectories, step_results = self.play_games_parallel(
                n_games=16, greedy_indices={0}
            )

            # 更新贪婪局（test）的 EMA 指标
            greedy_agent = agents[0]
            state = read_play_state()
            alpha = 0.001
            m = state["metrics"]
            m["test_piececount"] = m.get("test_piececount", 0) * (1 - alpha) + greedy_agent.piececount * alpha
            m["test_removedlines"] = m.get("test_removedlines", 0) * (1 - alpha) + greedy_agent.removedlines * alpha
            m["test_steps"] = m.get("test_steps", 0) * (1 - alpha) + greedy_agent.steps * alpha
            # 更新最高记录
            if greedy_agent.piececount > m.get("test_piececount_best", 0):
                m["test_piececount_best"] = greedy_agent.piececount
            if greedy_agent.removedlines > m.get("test_removedlines_best", 0):
                m["test_removedlines_best"] = greedy_agent.removedlines

            # 所有探索局（game 1-15）用于训练
            group_agents = [(agents[i], trajectories[i], step_results[i]) for i in range(1, len(agents))]

            # 更新 PPO 探索局 EMA 指标
            g_avg_pc = sum(a.piececount for a, _, _ in group_agents) / len(group_agents)
            g_avg_rl = sum(a.removedlines for a, _, _ in group_agents) / len(group_agents)
            g_avg_st = sum(a.steps for a, _, _ in group_agents) / len(group_agents)
            g_min_pc = min(a.piececount for a, _, _ in group_agents)
            g_max_pc = max(a.piececount for a, _, _ in group_agents)
            g_max_rl = max(a.removedlines for a, _, _ in group_agents)

            state["counters"]["agent"] += 1
            state["counters"]["_agent"] += 1

            m = state["metrics"]
            # PPO player EMA（带噪声探索的移动平均）
            m["ppo_piececount"]       = m.get("ppo_piececount",       0) * (1 - alpha) + g_avg_pc * alpha
            m["ppo_removedlines"]     = m.get("ppo_removedlines",     0) * (1 - alpha) + g_avg_rl * alpha
            m["ppo_steps"]            = m.get("ppo_steps",            0) * (1 - alpha) + g_avg_st * alpha
            m["ppo_piececount_min"]   = m.get("ppo_piececount_min",   9) * (1 - alpha) + g_min_pc * alpha
            m["ppo_piececount_max"]   = m.get("ppo_piececount_max",   0) * (1 - alpha) + g_max_pc * alpha
            # 历史最值
            m["ppo_piececount_best"]    = max(m.get("ppo_piececount_best",    0), g_max_pc)
            m["ppo_removedlines_best"]  = max(m.get("ppo_removedlines_best",  0), g_max_rl)

            print(f"Group: ppo_avg={g_avg_pc:.1f} min={g_min_pc} max={g_max_pc} lines_avg={g_avg_rl:.2f}")

            save_play_state(state)

            # 检查是否刷新历史最佳（按ppo EMA方块数）
            ppo_ema_pc = m.get("ppo_piececount", 0)
            old_best_pc = m.get("ppo_piececount_best_ema", 0)
            if ppo_ema_pc > old_best_pc:
                m["ppo_piececount_best_ema"] = ppo_ema_pc
                save_play_state(state)  # 保存更新后的 best_ema
                # 保存最佳模型（按ppo EMA方块数建目录，备份状态文件）
                best_dir = os.path.join(model_dir, f"{ppo_ema_pc:.1f}")
                os.makedirs(best_dir, exist_ok=True)
                self.policy_net.save_model(os.path.join(best_dir, 'model.pth'))
                shutil.copy2(play_file, os.path.join(best_dir, 'play.json'))
                if os.path.exists(train_file):
                    shutil.copy2(train_file, os.path.join(best_dir, 'train.json'))
                print(f"*** new best! ppo_ema={ppo_ema_pc:.1f} > best={old_best_pc:.1f}, saved to {best_dir}")


            # 保存每局结果：一局一个 pkl 文件（包含所有 step）
            filetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for run_idx, (agent, trajectory, step_results) in enumerate(group_agents):
                game_counter += 1

                # 每步存储: (state, ref_prob, log_prob, action, prev_action, r_step, is_terminal, availables)
                n_steps = len(trajectory)
                # 全局效率奖励：piececount / steps（鼓励高效放置）
                # default_r = agent.piececount / max(agent.steps, 1)
                game_steps = []
                for step_idx, step_data in enumerate(trajectory):
                    landed, removed = step_results[step_idx]
                    is_terminal = 1 if step_idx == n_steps - 1 else 0
                    r_step = 0
                    # if landed:
                    #     r_step = -0.1                         # 落地惩罚
                    #     if removed == 1:   r_step = 0.25       # 消1行
                    #     elif removed == 2: r_step = 0.5        # 消2行
                    #     elif removed == 3: r_step = 0.75       # 消3行
                    #     elif removed >= 4: r_step = 1.0        # 消4行（Tetris）
                    if is_terminal:
                        r_step = -1 # default_r
                    
                    game_steps.append((
                        step_data["state"], step_data["ref_prob"],
                        step_data["log_prob"], step_data["action"],
                        step_data["prev_action"], r_step, is_terminal,
                        step_data["availables"]
                    ))
                    
                filename = f"{filetime}-{game_counter:06d}-r{run_idx}.pkl"
                savefile = os.path.join(data_wait_dir, filename)
                with open(savefile, "wb") as fn:
                    pickle.dump(game_steps, fn)

            # 更新计数器 + 历史统计（用实际游戏数据，保证 show_status 有数据）
            alpha = 0.001
            g_avg_pc = sum(a.piececount for a, _, _ in group_agents) / len(group_agents)
            g_avg_rl = sum(a.removedlines for a, _, _ in group_agents) / len(group_agents)
            g_avg_st = sum(a.steps for a, _, _ in group_agents) / len(group_agents)
            g_min_pc = min(a.piececount for a, _, _ in group_agents)
            g_max_pc = max(a.piececount for a, _, _ in group_agents)
            g_max_rl = max(a.removedlines for a, _, _ in group_agents)

            state = read_play_state()
            state["counters"]["agent"] += 1
            state["counters"]["_agent"] += 1

            m = state["metrics"]
            # PPO player EMA（带噪声探索的移动平均）
            m["ppo_piececount"]       = m.get("ppo_piececount",       0) * (1 - alpha) + g_avg_pc * alpha
            m["ppo_removedlines"]     = m.get("ppo_removedlines",     0) * (1 - alpha) + g_avg_rl * alpha
            m["ppo_steps"]            = m.get("ppo_steps",            0) * (1 - alpha) + g_avg_st * alpha
            m["ppo_piececount_min"]   = m.get("ppo_piececount_min",   9) * (1 - alpha) + g_min_pc * alpha
            m["ppo_piececount_max"]   = m.get("ppo_piececount_max",   0) * (1 - alpha) + g_max_pc * alpha
            # 历史最值
            m["ppo_piececount_best"]    = max(m.get("ppo_piececount_best",    0), g_max_pc)
            m["ppo_removedlines_best"]  = max(m.get("ppo_removedlines_best",  0), g_max_rl)

            save_play_state(state)

        print(f"\nCollection finished. Total games: {game_counter}")

    def run(self):
        """运行数据采集"""
        try:
            self.collect_ppo_data()                
        except KeyboardInterrupt:
            print('quit')


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print('start PPO selfplay', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    np.set_printoptions(precision=2, suppress=True)
    training = PPOSelfPlay()
    training.run()
    print('end PPO selfplay', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
