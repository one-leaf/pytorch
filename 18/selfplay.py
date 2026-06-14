import os, pickle, time, random, itertools
from datetime import datetime
import numpy as np
import torch

from model import PolicyValueNet, data_dir, data_wait_dir, model_file
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
        self._test_policy_value_net = None
        self.policy_value_net = None

    def get_action_from_policy(self, agent, policy_value_net, train=True):
        """从策略网络采样一个动作（带动作掩码）"""
        state = np.array([agent.current_state()])
        device = policy_value_net.device
        state_tensor = torch.FloatTensor(state).to(device)

        policy_value_net.policy_value_net.eval()
        with torch.no_grad():
            log_probs, _ = policy_value_net.policy_value_net(state_tensor)
        probs = torch.exp(log_probs[0]).cpu().numpy()  # [5]

        if train:
            p = 0.75
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

    def play_one_game(self, isRandomNextPiece=True, nextPiecesList=None, train=True):
        """用当前策略玩一局游戏，记录完整轨迹"""
        if nextPiecesList is not None and len(nextPiecesList) > 0:
            agent = Agent(isRandomNextPiece=False, nextPiecesList=nextPiecesList)
        else:
            agent = Agent(isRandomNextPiece=isRandomNextPiece)

        trajectory = []  # [(state, action, ref_prob, log_prob), ...]

        for _ in range(self.rollout_max_steps):
            if agent.terminal:
                break

            state = agent.current_state().copy()
            action, probs, log_prob = self.get_action_from_policy(agent, self.policy_value_net, train)

            trajectory.append({
                "state": state,
                "action": action,
                "ref_prob": probs.copy(),
                "log_prob": log_prob.copy(),
                "piececount": agent.piececount,
                "score": agent.piececount,
            })

            _, reward = agent.step(action)

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

        for _ in range(test_count):
            agent = Agent(isRandomNextPiece=True)
            for i in range(self.max_step_count):
                action, _, _ = self.get_action_from_policy(
                    agent, self._test_policy_value_net
                )
                _, reward = agent.step(action)
                if agent.terminal:
                    break

            agent.print()

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

        print(f"test: min_pieces={min_pieces_count} max_pieces={max_pieces_count} "
              f"min_lines={min_removedlines} max_lines={max_removedlines}")

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
                best_removedlines, worst_removedlines)

    def collect_grpo_data(self):
        """收集 GRPO 自我对抗数据"""
        print("GRPO Self Play starting ...")

        load_model_file = model_file
        if os.path.exists(model_file + "_best"):
            load_model_file = model_file + "_best"
        elif os.path.exists(model_file + ".bak"):
            load_model_file = model_file + ".bak"

        if os.path.exists(load_model_file):
            if time.time() - os.path.getmtime(load_model_file) > 60 * 60 * 5:
                print("超过5小时模型都没有更新了，停止训练")
                time.sleep(60)
                return

        print('start test time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if self._test_policy_value_net is None:
            self._test_policy_value_net = PolicyValueNet(
                GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=load_model_file
            )
        (_, his_pieces, his_pieces_len, _,
         max_pieces_count, min_pieces_count,
         best_removedlines, worst_removedlines) = self.test_play()
        print('end test time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # 加载模型用于数据收集
        if self.policy_value_net is None:
            self.policy_value_net = PolicyValueNet(
                GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=load_model_file
            )

        # 检查重玩列表
        replay_dir = os.path.join(data_dir, "replay")
        if not os.path.exists(replay_dir):
            os.makedirs(replay_dir)
        listFiles = [f for f in os.listdir(replay_dir) if f.endswith(".pkl")]
        if listFiles and random.random() > 0.20:
            earliest_files = sorted(listFiles, key=lambda f: os.path.getctime(os.path.join(replay_dir, f)))
            filename = os.path.join(replay_dir, earliest_files[0])
            try:
                with open(filename, "rb") as fn:
                    his_pieces = pickle.load(fn)
                    his_pieces_len = len(his_pieces)
                print(f"load need replay {filename}")
            finally:
                os.remove(filename)
        else:
            his_pieces = []
            his_pieces_len = 0

        # 读取历史平均步数作为最小步数阈值
        state = read_status_file()
        avg_steps = state["metrics"]["grpo_steps"]
        if avg_steps < 1:
            avg_steps = 100  # 首次无历史数据时的默认值

        # 持续采集，每局完成后立即保存
        print("starting continuous collection ...")
        _start_time = time.time()
        game_counter = 0

        for g in itertools.count():
            if time.time() - _start_time > 60 * 20:  # 每个模型最多20分钟采集
                break

            # 确定本局方块序列
            if g == 0 and his_pieces_len > 0:
                pieces_list = his_pieces
            else:
                agent0, _ = self.play_one_game(isRandomNextPiece=True)
                pieces_list = agent0.piecehis

            # 用相同方块序列运行 8 局
            group_agents = []
            # group_rewards = []
            for _ in range(8):
                agent, trajectory = self.play_one_game(
                    isRandomNextPiece=False, nextPiecesList=pieces_list,
                )
                if len(trajectory) > 0:
                    group_agents.append((agent, trajectory))
                agent.print()  
                if len(agent.piecehis)>len(pieces_list):
                    pieces_list = agent.piecehis
                
            if len(group_agents) == 0:
                continue

            # 奖励设计：消行权重压倒 piececount，鼓励冒险消行
            raw_rewards = [agent.piececount for agent, _ in group_agents]
            raw_arr = np.array(raw_rewards)

            # 组内正规化
            mean_r = raw_arr.mean()
            std_r = raw_arr.std() + 1e-6
            norm_rewards = (raw_arr - mean_r) / std_r

            # 打印信息
            print(f"Group {g}: raw_rewards={raw_arr}  mean={mean_r:.3f} std={std_r:.3f}")

            # 保存每局结果
            filetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for run_idx, (agent, trajectory) in enumerate(group_agents):
                norm_reward = norm_rewards[run_idx]
                game_counter += 1
                print(f"  Run {run_idx + 1}: raw={raw_rewards[run_idx]:.3f} norm={norm_reward:.4f} "
                      f"piececount={agent.piececount} removedlines={agent.removedlines} steps={agent.steps}")

                # 保存该局的 step 数据（已正规化的 reward）
                for i, step_data in enumerate(trajectory):
                    data = (step_data["state"], step_data["ref_prob"], float(norm_reward), step_data["action"], 1)
                    filename = f"{filetime}-{game_counter:06d}-r{run_idx}-{i}.pkl"
                    savefile = os.path.join(data_wait_dir, filename)
                    with open(savefile, "wb") as fn:
                        pickle.dump(data, fn)

            # 一组完成后，更新训练状态
            state = read_status_file()
            state["counters"]["agent"] += 1
            state["counters"]["_agent"] += 1
            
            avg_pc = np.mean([a.piececount for a, _ in group_agents])
            avg_rl = np.mean([a.removedlines for a, _ in group_agents])
            avg_st = np.mean([a.steps for a, _ in group_agents])

            state["_accum"]["_sum_piececount"] += avg_pc
            state["_accum"]["_sum_removedlines"] += avg_rl
            state["_accum"]["_sum_steps"] += avg_st

            # test_play 历史最值
            state["metrics"]["grpo_removedlines_best"] = max(state["metrics"]["grpo_removedlines_best"], best_removedlines)
            state["metrics"]["grpo_piececount_best"] = max(state["metrics"]["grpo_piececount_best"], max_pieces_count)
            state["metrics"]["grpo_removedlines_worst"] = min(state["metrics"]["grpo_removedlines_worst"], worst_removedlines)
            state["metrics"]["grpo_piececount_worst"] = min(state["metrics"]["grpo_piececount_worst"], min_pieces_count)

            # EMA 移动平均（用组平均）
            alpha = 0.1
            old_pc = state["metrics"].get("grpo_piececount", 0)
            old_rl = state["metrics"].get("grpo_removedlines", 0)
            old_st = state["metrics"].get("grpo_steps", 0)
            old_reward_mean = state["metrics"].get("grpo_reward_mean", 0)
            old_reward_std = state["metrics"].get("grpo_reward_std", 0)
            state["metrics"]["grpo_piececount"] = round(old_pc * (1 - alpha) + avg_pc * alpha, 3)
            state["metrics"]["grpo_removedlines"] = round(old_rl * (1 - alpha) + avg_rl * alpha, 3)
            state["metrics"]["grpo_steps"] = round(old_st * (1 - alpha) + avg_st * alpha, 3)
            state["metrics"]["grpo_reward_mean"] = round(old_reward_mean * (1 - alpha) + mean_r * alpha, 3)
            state["metrics"]["grpo_reward_std"] = round(old_reward_std * (1 - alpha) + std_r * alpha, 3)

            # 组内最值
            state["metrics"]["grpo_piececount_min"] = min(state["metrics"]["grpo_piececount_min"], min(a.piececount for a, _ in group_agents))
            state["metrics"]["grpo_piececount_max"] = max(state["metrics"]["grpo_piececount_max"], max(a.piececount for a, _ in group_agents))
            state["metrics"]["grpo_removedlines_min"] = min(state["metrics"]["grpo_removedlines_min"], min(a.removedlines for a, _ in group_agents))
            state["metrics"]["grpo_removedlines_max"] = max(state["metrics"]["grpo_removedlines_max"], max(a.removedlines for a, _ in group_agents))
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
