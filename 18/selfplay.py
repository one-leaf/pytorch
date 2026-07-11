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


class GRPOSelfPlay():
    def __init__(self):
        self.rollout_max_steps = 500    # 单局最大步数
        self.test_count = 10            # 测试次数
        self.max_step_count = 10000     # 最大步数限制
        self.policy_net = None

    def get_action_from_policy(self, agent, policy_net, prev_action, train=True):
        """从策略网络采样一个动作（带动作掩码）"""
        state = np.array([agent.current_state()])
        device = policy_net.device
        state_tensor = torch.FloatTensor(state).to(device)
        prev_action_tensor = torch.LongTensor([prev_action]).to(device)

        policy_net.net.eval()
        with torch.no_grad():
            log_probs, _ = policy_net.net(state_tensor, prev_action_tensor)
        if torch.isnan(log_probs).any():
            log_probs = torch.zeros_like(log_probs)
        probs = torch.exp(log_probs[0]).cpu().numpy()  # [5]

        if train:
            p = 0.98
            dirichlet = np.random.dirichlet(2 * np.ones(GAME_ACTIONS_NUM))            
            probs = p*probs + (1.0-p)*dirichlet
            probs = probs / np.sum(probs) 

            # 应用动作掩码
            availables = agent.availables  # [5] 0/1 掩码
            probs = probs * availables.astype(np.float32)
            probs_sum = probs.sum()
            if probs_sum < 1e-10:
                probs = availables.astype(np.float32)
                probs_sum = probs.sum()
            probs = probs / probs_sum

            action = np.random.choice(GAME_ACTIONS_NUM, p=probs)
        else:
            availables = agent.availables  # [5] 0/1 掩码
            probs = probs * availables.astype(np.float32)
            action = np.argmax(probs)
            
        return int(action), probs, log_probs[0].cpu().numpy()

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

    def play_one_game(self, isRandomNextPiece=True, nextPiecesList=None, train=True):
        """用当前策略玩一局游戏，记录完整轨迹"""
        if nextPiecesList is not None and len(nextPiecesList) > 0:
            agent = Agent(isRandomNextPiece=False, nextPiecesList=nextPiecesList)
        else:
            agent = Agent(isRandomNextPiece=isRandomNextPiece)

        trajectory = []
        prev_action = 3  # KEY_NONE

        for _ in range(self.rollout_max_steps):
            if agent.terminal:
                break

            state = agent.current_state().copy()
            action, probs, log_prob = self.get_action_from_policy(agent, self.policy_net, prev_action, train)

            trajectory.append({
                "state": state,
                "action": action,
                "prev_action": prev_action,
                "ref_prob": probs.copy(),
                "log_prob": log_prob.copy(),
            })

            prev_action = action
            agent.step(action)

        return agent, trajectory

    def test_play(self, test_count=None):
        """测试模式：贪婪策略评估"""
        if test_count is None:
            test_count = self.test_count

        min_pieces_count = 999999
        max_pieces_count = 0
        min_removedlines = 0
        max_removedlines = 0
        best_removedlines = 0     # 测试中最多消除行数
        worst_removedlines = 999999  # 测试中最少消除行数
        min_his_pieces = None
        min_his_pieces_len = 0
        sum_piececount = 0
        sum_removedlines = 0
        sum_steps = 0
        test_games = 0

        for _ in range(test_count):
            agent = Agent(isRandomNextPiece=True)
            prev_action = 3  # KEY_NONE
            for _ in range(self.max_step_count):
                action, _, _ = self.get_action_from_policy(
                    agent, self.policy_net, prev_action, train=False
                )
                prev_action = action
                agent.step(action)
                if agent.terminal:
                    break

            agent.print()
            sum_piececount += agent.piececount
            sum_removedlines += agent.removedlines
            sum_steps += agent.steps
            test_games += 1

            # 跟踪最差/最好局面
            if agent.piececount < min_pieces_count:
                min_pieces_count = agent.piececount
                min_his_pieces = agent.piecehis
                min_his_pieces_len = len(agent.piecehis)
                min_removedlines = agent.removedlines
            if agent.piececount > max_pieces_count:
                max_pieces_count = agent.piececount
                max_removedlines = agent.removedlines

            # 独立跟踪消除行数最值
            if agent.removedlines > best_removedlines:
                best_removedlines = agent.removedlines
            if agent.removedlines < worst_removedlines:
                worst_removedlines = agent.removedlines

        avg_pc = sum_piececount / max(test_games, 1)
        avg_rl = sum_removedlines / max(test_games, 1)
        avg_st = sum_steps / max(test_games, 1)
        print(f"test: min_pieces={min_pieces_count} max_pieces={max_pieces_count} "
              f"min_lines={min_removedlines} max_lines={max_removedlines} "
              f"avg_pieces={avg_pc:.1f} avg_lines={avg_rl:.3f} avg_steps={avg_st:.1f}")

        # 保存最差局面用于重玩
        if min_pieces_count < 20 and min_his_pieces:
            replay_dir = os.path.join(data_dir, "replay")
            if not os.path.exists(replay_dir):
                os.makedirs(replay_dir)
            filename = f"{min_his_pieces_len:05d}-{min_removedlines:05d}-{''.join(min_his_pieces)[:50]}.pkl"
            his_pieces_file = os.path.join(replay_dir, filename)
            print(f"save need replay {his_pieces_file}")
            with open(his_pieces_file, "wb") as fn:
                pickle.dump(min_his_pieces, fn)

        return (min_removedlines, min_his_pieces, min_his_pieces_len,
                max_removedlines, max_pieces_count, min_pieces_count,
                best_removedlines, worst_removedlines,
                avg_pc, avg_rl, avg_st)

    def collect_grpo_data(self):
        """收集 GRPO 自我对抗数据"""
        print("GRPO Self Play starting ...")

        # 确定初始模型文件
        load_model_file = model_file
        if os.path.exists(model_file + "_best"):
            load_model_file = model_file + "_best"
        elif os.path.exists(model_file + ".bak"):
            load_model_file = model_file + ".bak"

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

        # 检查重玩列表
        replay_dir = os.path.join(data_dir, "replay")
        if not os.path.exists(replay_dir):
            os.makedirs(replay_dir)
        his_pieces = []
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

        # 持续采集，每局完成后立即保存
        print("starting continuous collection ...")
        _start_time = time.time()
        game_counter = 0

        for g in itertools.count():
            if time.time() - _start_time > 60 * 20:  # 每个模型最多20分钟采集
                break

            # 每组之前检查模型是否有更新，有则重新加载
            current_model = model_file
            if os.path.exists(model_file + "_best"):
                current_model = model_file + "_best"
            elif os.path.exists(model_file + ".bak"):
                current_model = model_file + ".bak"
            if os.path.exists(current_model):
                mtime = os.path.getmtime(current_model)
                if mtime > _last_model_mtime:
                    print(f"Model updated, reloading from {current_model}")
                    self.policy_net = PolicyNet(
                        GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=current_model
                    )
                    self._check_and_fix_nan(self.policy_net)
                    _last_model_mtime = mtime

            # 确定本局方块序列
            if g == 0 and len(his_pieces) > 0:
                pieces_list = his_pieces
                his_pieces = []
            else:
                agent0, _ = self.play_one_game(isRandomNextPiece=True)
                pieces_list = agent0.piecehis

            # 用相同方块序列运行 4 局
            group_pieces_list = pieces_list  # 组内共享同一序列
            group_agents = []
            for _ in range(4):
                agent, trajectory = self.play_one_game(
                    isRandomNextPiece=False, nextPiecesList=group_pieces_list,
                )
                if len(trajectory) > 0:
                    group_agents.append((agent, trajectory))
                if len(agent.piecehis) > len(pieces_list):
                    pieces_list = agent.piecehis  # 只在组外使用，不影响本组内其他局

            if len(group_agents) == 0:
                continue

            # 只保留最差和最好的 2 局（≥3 局时筛选）
            if len(group_agents) >= 3:
                pcs = [a.piececount for a, _ in group_agents]
                best_idx = max(range(len(pcs)), key=lambda i: pcs[i])
                worst_idx = min(range(len(pcs)), key=lambda i: pcs[i])
                keep = sorted(set([best_idx, worst_idx]))
                group_agents = [group_agents[i] for i in keep]

            # 游戏级基础奖励：piececount（消行信息已编码在 piececount 差异中）
            N_arr = np.array([agent.piececount for agent, _ in group_agents])
            raw_rewards = N_arr.copy()

            # 组内正规化
            mean_r = raw_rewards.mean()
            std_r = raw_rewards.std() + 1e-6
            norm_rewards = (raw_rewards - mean_r) / std_r

            # 打印信息
            print(f"Group {g}: piececounts={N_arr}  mean={mean_r:.3f} std={std_r:.3f}")

            # 保存每局结果：一局一个 pkl 文件（包含所有 step）
            filetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for run_idx, (agent, trajectory) in enumerate(group_agents):
                game_counter += 1
                R = float(agent.piececount)  # 该局的目标回报

                # 每步存储: (state, ref_prob, log_prob, action, prev_action, game_id, R)
                game_steps = [
                    (step_data["state"], step_data["ref_prob"],
                     step_data["log_prob"], step_data["action"],
                     step_data["prev_action"], game_counter, R)
                    for step_data in trajectory
                ]
                filename = f"{filetime}-{game_counter:06d}-r{run_idx}.pkl"
                savefile = os.path.join(data_wait_dir, filename)
                with open(savefile, "wb") as fn:
                    pickle.dump(game_steps, fn)

                print(f"  Run {run_idx + 1}: raw={raw_rewards[run_idx]:.1f} norm={norm_rewards[run_idx]:.4f} "
                      f"piececount={agent.piececount} removedlines={agent.removedlines} steps={agent.steps}")

            # 更新计数器 + 历史统计（用实际游戏数据，保证 show_status 有数据）
            alpha = 0.1
            g_avg_pc = sum(a.piececount for a, _ in group_agents) / len(group_agents)
            g_avg_rl = sum(a.removedlines for a, _ in group_agents) / len(group_agents)
            g_avg_st = sum(a.steps for a, _ in group_agents) / len(group_agents)
            g_min_pc = min(a.piececount for a, _ in group_agents)
            g_max_pc = max(a.piececount for a, _ in group_agents)
            g_max_rl = max(a.removedlines for a, _ in group_agents)

            state = read_status_file()
            state["counters"]["agent"] += 1
            state["counters"]["_agent"] += 1

            state["_accum"]["_sum_piececount"]   += g_avg_pc
            state["_accum"]["_sum_removedlines"] += g_avg_rl
            state["_accum"]["_sum_steps"]        += g_avg_st

            m = state["metrics"]
            # GRPO player EMA（带噪声探索的移动平均）
            m["grpo_piececount"]       = round(m.get("grpo_piececount",       0) * (1 - alpha) + g_avg_pc * alpha, 3)
            m["grpo_removedlines"]     = round(m.get("grpo_removedlines",     0) * (1 - alpha) + g_avg_rl * alpha, 3)
            m["grpo_steps"]            = round(m.get("grpo_steps",            0) * (1 - alpha) + g_avg_st * alpha, 3)
            m["grpo_piececount_min"]   = round(m.get("grpo_piececount_min",   999999) * (1 - alpha) + g_min_pc * alpha, 3)
            m["grpo_piececount_max"]   = round(m.get("grpo_piececount_max",   0)      * (1 - alpha) + g_max_pc * alpha, 3)
            # 历史最值
            m["grpo_piececount_best"]    = max(m.get("grpo_piececount_best",    0), g_max_pc)
            m["grpo_removedlines_best"]  = max(m.get("grpo_removedlines_best",  0), g_max_rl)

            save_status_file(state)
            print(f"status updated: agent={state['counters']['agent']}")

        print(f"\nCollection finished. Total games: {game_counter}")

    def run(self):
        """运行数据采集"""
        try:
            self.collect_grpo_data()                
        except KeyboardInterrupt:
            print('quit')


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print('start GRPO selfplay', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    np.set_printoptions(precision=2, suppress=True)
    training = GRPOSelfPlay()
    training.run()
    print('end GRPO selfplay', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
