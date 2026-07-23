import os, pickle, time, random, itertools
from datetime import datetime
import numpy as np
import torch

from model import PolicyNet, data_dir, data_wait_dir, model_file
from agent import Agent, ACTIONS
from status import save_status_file, read_status_file

# 定义游戏的动作
GAME_ACTIONS_NUM = len(ACTIONS)
GAME_WIDTH, GAME_HEIGHT = 10, 20


class PPOSelfPlay():
    def __init__(self):
        self.rollout_max_steps = 500    # 单局最大步数
        self.policy_net = None

    def get_actions_batch(self, agents, prev_actions, temperature=1.0, greedy_indices=None):
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

        for idx, i in enumerate(active_indices):
            agent = agents[i]
            log_probs = log_probs_batch[idx]
            probs = torch.exp(log_probs / temperature).cpu().numpy()

            if i not in greedy_indices:
                p = 0.98
                dirichlet = np.random.dirichlet(0.3 * np.ones(GAME_ACTIONS_NUM))
                probs = p * probs + (1.0 - p) * dirichlet
                probs = probs / np.sum(probs)

                availables = agent.availables
                probs = probs * availables.astype(np.float32)
                probs_sum = probs.sum()
                if probs_sum < 1e-10:
                    probs = availables.astype(np.float32)
                    probs_sum = probs.sum()
                probs = probs / probs_sum
                action = np.random.choice(GAME_ACTIONS_NUM, p=probs)
            else:
                availables = agent.availables
                probs = probs * availables.astype(np.float32)
                action = np.argmax(probs)

            actions.append(int(action))
            all_probs.append(probs.copy())
            all_log_probs.append(log_probs.cpu().numpy())

        return actions, all_probs, all_log_probs

    def play_games_parallel(self, n_games=4, pieces_list=None, temperature=1.0, greedy_indices=None):
        """同时玩 n_games 局，共享方块序列，批量预测"""
        agents = [Agent(isRandomNextPiece=False, nextPiecesList=pieces_list) for _ in range(n_games)]
        trajectories = [[] for _ in range(n_games)]
        step_results = [[] for _ in range(n_games)]
        prev_actions = [3] * n_games  # KEY_NONE

        for _ in range(self.rollout_max_steps):
            if all(a.terminal for a in agents):
                break

            actions, all_probs, all_log_probs = self.get_actions_batch(
                agents, prev_actions, temperature, greedy_indices
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
                action_idx += 1

                trajectories[i].append({
                    "state": state,
                    "action": action,
                    "prev_action": prev_actions[i],
                    "ref_prob": probs,
                    "log_prob": log_prob,
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
        # if os.path.exists(model_file + "_best"):
        #     load_model_file = model_file + "_best"
        # elif os.path.exists(model_file + ".bak"):
        #     load_model_file = model_file + ".bak"

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

        # 重玩目录
        replay_dir = os.path.join(data_dir, "replay")
        if not os.path.exists(replay_dir):
            os.makedirs(replay_dir)
        his_pieces = []

        # 持续采集，每局完成后立即保存
        print("starting continuous collection ...")
        _start_time = time.time()
        game_counter = 0

        for g in itertools.count():
            if time.time() - _start_time > 60 * 60:  # 最多60分钟采集
                break

            # 每组之前检查模型是否有更新，有则重新加载
            current_model = model_file
            # if os.path.exists(model_file + "_best"):
            #     current_model = model_file + "_best"
            # elif os.path.exists(model_file + ".bak"):
            #     current_model = model_file + ".bak"
            if os.path.exists(current_model):
                mtime = os.path.getmtime(current_model)
                if mtime > _last_model_mtime:
                    print(f"Model updated, reloading from {current_model}")
                    self.policy_net = PolicyNet(
                        GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=current_model
                    )
                    self._check_and_fix_nan(self.policy_net)
                    _last_model_mtime = mtime

            # 每 10 轮检查一次重玩数据
            his_pieces = []
            if g % 10 == 0:
                listFiles = [f for f in os.listdir(replay_dir) if f.endswith(".pkl")]
                if listFiles and random.random() > 0.20:
                    earliest_files = sorted(listFiles, key=lambda f: os.path.getctime(os.path.join(replay_dir, f)))
                    filename = os.path.join(replay_dir, earliest_files[0])
                    try:
                        with open(filename, "rb") as fn:
                            his_pieces = pickle.load(fn)
                        print(f"load need replay {filename}")
                    finally:
                        os.remove(filename)

            # 确定本局方块序列：有重玩数据则用重玩，否则随机生成
            if len(his_pieces) > 0:
                pieces_list = his_pieces
            else:
                status = read_status_file()
                avg_pc = status["metrics"].get("ppo_piececount", 50)
                pieces_list = [random.choice(['s', 'z', 'i', 'o', 'l', 'j', 't']) for _ in range(int(avg_pc))]                
            pieces_list += [random.choice(['s', 'z', 'i', 'o', 'l', 'j', 't']) for _ in range(1000)]    

            # 并行玩 8 局，取 max 和 min
            agents, trajectories, step_results = self.play_games_parallel(
                n_games=8, pieces_list=pieces_list, temperature=1.0, greedy_indices={0}
            )

            pcs = [a.piececount for a in agents]

            # 更新贪婪局（test）的 EMA 指标
            greedy_agent = agents[0]
            state = read_status_file()
            alpha = 0.001
            m = state["metrics"]
            m["test_piececount"] = m.get("test_piececount", 0) * (1 - alpha) + greedy_agent.piececount * alpha
            m["test_removedlines"] = m.get("test_removedlines", 0) * (1 - alpha) + greedy_agent.removedlines * alpha
            m["test_steps"] = m.get("test_steps", 0) * (1 - alpha) + greedy_agent.steps * alpha

            # 检查是否刷新历史最佳
            old_best_pc = m.get("test_piececount_best", 0)
            if greedy_agent.piececount > old_best_pc:
                m["test_piececount_best"] = greedy_agent.piececount
                m["test_removedlines_best"] = max(m.get("test_removedlines_best", 0), greedy_agent.removedlines)
                # 保存最佳模型
                best_model_path = f"{model_file}.{greedy_agent.piececount:.1f}"
                self.policy_net.save_model(best_model_path)
                self.policy_net.save_model(model_file + ".bak")
                print(f"*** new best! greedy_piececount={greedy_agent.piececount} > best={old_best_pc}, saved to {best_model_path}")

            save_status_file(state)

            # 判断是否保留：最大最小相差 2 以上则保存全部 4 局
            if max(pcs) - min(pcs) >= 2:
                group_agents = [(agents[i], trajectories[i], step_results[i]) for i in range(len(agents))]
            else:
                print(f"Group {g}: piececounts={pcs}, diff={max(pcs) - min(pcs)} < 2, skipping")
                # 导出最佳局的历史方块到重玩目录
                best_idx = max(range(len(pcs)), key=lambda i: pcs[i])
                best_pieces = agents[best_idx].piecehis
                if len(best_pieces) > 10:
                    filename = f"{len(best_pieces):05d}-{agents[best_idx].removedlines:05d}-{''.join(best_pieces)[:50]}.pkl"
                    savefile = os.path.join(replay_dir, filename)
                    with open(savefile, "wb") as fn:
                        pickle.dump(best_pieces, fn)
                continue

            # 游戏级奖励：piececount + 消行奖励
            N_arr = np.array([agent.piececount for agent, _, _ in group_agents])
            L_arr = np.array([agent.removedlines for agent, _, _ in group_agents])

            # 打印信息
            print(f"Group {g}: all_piececounts={pcs} selected={N_arr} lines={L_arr}")

            # 保存每局结果：一局一个 pkl 文件（包含所有 step）
            filetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for run_idx, (agent, trajectory, step_results) in enumerate(group_agents):
                game_counter += 1

                # 每步存储: (state, ref_prob, log_prob, action, prev_action, r_step, is_terminal)
                # r_step: 中间步 落地未消行 -0.1，落地消行 +0.1；最后一步按方块数给奖励
                n_steps = len(trajectory)
                game_steps = []
                for step_idx, step_data in enumerate(trajectory):
                    landed, removed = step_results[step_idx]
                    is_terminal = 1 if step_idx == n_steps - 1 else 0
                    if is_terminal:
                        r_step = float(agent.piececount)  # 最后一步：按方块数
                    else:
                        r_step = 0.0
                        if landed:
                            r_step = 0.0 if removed == 0 else 0.1
                    game_steps.append((
                        step_data["state"], step_data["ref_prob"],
                        step_data["log_prob"], step_data["action"],
                        step_data["prev_action"], r_step, is_terminal
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

            state = read_status_file()
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

            save_status_file(state)

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
