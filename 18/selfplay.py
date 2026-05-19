import os, glob, pickle, time, random
from datetime import datetime
import numpy as np
import torch

from model import PolicyValueNet, data_dir, data_wait_dir, model_file
from agent import Agent, ACTIONS
from status import save_status_file, read_status_file, set_status_total_value
from augment import get_equi_data

# 定义游戏的动作
GAME_ACTIONS_NUM = len(ACTIONS)
GAME_WIDTH, GAME_HEIGHT = 10, 20


class GRPOSelfPlay():
    def __init__(self):
        self.num_games = 8              # GRPO 组大小 G
        self.rollout_max_steps = 500    # 单局最大步数
        self.test_count = 10            # 测试次数
        self.max_step_count = 10000     # 最大步数限制

    def get_action_from_policy(self, policy_value_net, agent, device):
        """从策略网络采样一个动作（带动作掩码）"""
        state = np.array([agent.current_state()])
        state_tensor = torch.FloatTensor(state).to(device)

        policy_value_net.policy_value_net.eval()
        with torch.no_grad():
            log_probs, _ = policy_value_net.policy_value_net(state_tensor)

        probs = torch.exp(log_probs[0]).cpu().numpy()  # [5]

        # 应用动作掩码
        availables = agent.availables  # [5] 0/1 掩码
        probs = probs * availables.astype(np.float32)
        probs_sum = probs.sum()
        if probs_sum < 1e-10:
            probs = availables.astype(np.float32)
            probs_sum = probs.sum()
        probs = probs / probs_sum

        action = np.random.choice(GAME_ACTIONS_NUM, p=probs)
        return int(action), probs, log_probs[0].cpu().numpy()

    def play_one_game(self, policy_value_net, device, isRandomNextPiece=True, nextPiecesList=None):
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
            action, probs, log_prob = self.get_action_from_policy(policy_value_net, agent, device)

            trajectory.append({
                "state": state,
                "action": action,
                "ref_prob": probs.copy(),
                "log_prob": log_prob.copy(),
                "piececount": agent.piececount,
                "score": agent.removedlines,
            })

            _, reward = agent.step(action)

        return agent, trajectory

    def test_play(self, policy_value_net, test_count=None):
        """测试模式：贪婪策略评估"""
        if test_count is None:
            test_count = self.test_count

        min_pieces_count = 999999
        max_pieces_count = 0
        min_removedlines = 0
        max_removedlines = 0
        min_his_pieces = None
        min_his_pieces_len = 0

        for _ in range(test_count):
            agent = Agent(isRandomNextPiece=True)
            for i in range(self.max_step_count):
                action, _, _ = self.get_action_from_policy(
                    policy_value_net, agent, policy_value_net.device
                )
                _, reward = agent.step(action)
                if agent.terminal:
                    break

            agent.print()

            if agent.piececount < min_pieces_count:
                min_pieces_count = agent.piececount
                min_his_pieces = agent.piecehis
                min_his_pieces_len = len(agent.piecehis)
                min_removedlines = agent.removedlines
            if agent.piececount > max_pieces_count:
                max_pieces_count = agent.piececount
                max_removedlines = agent.removedlines

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

        return min_removedlines, min_his_pieces, min_his_pieces_len, max_removedlines, max_pieces_count, min_pieces_count

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
        _test_policy_value_net = PolicyValueNet(
            GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=load_model_file
        )
        min_removedlines, his_pieces, his_pieces_len, max_removedlines, max_pieces_count, min_pieces_count = self.test_play(_test_policy_value_net)
        print('end test time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # 加载模型用于数据收集
        policy_value_net = PolicyValueNet(
            GAME_WIDTH, GAME_HEIGHT, GAME_ACTIONS_NUM, model_file=load_model_file
        )
        device = policy_value_net.device

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

        G = self.num_games
        print(f"GRPO group size G={G}")

        all_states = []
        all_ref_probs = []
        all_advantages = []
        all_actions = []
        all_masks = []

        total_score = 0
        total_piececount = 0
        total_steps = 0
        min_piececount = 999999
        max_piececount = 0

        # 运行 G 局游戏
        for g in range(G):
            print(f"\n=== Game {g + 1}/{G} ===")
            use_replay = (g == 0 and his_pieces_len > 0)
            agent, trajectory = self.play_one_game(
                policy_value_net, device,
                isRandomNextPiece=not use_replay,
                nextPiecesList=his_pieces if use_replay else None
            )

            total_reward = agent.removedlines
            print(f"Game {g}: removedlines={total_reward}, piececount={agent.piececount}, steps={agent.steps}")

            total_score += total_reward
            total_piececount += agent.piececount
            total_steps += agent.steps
            if agent.piececount < min_piececount:
                min_piececount = agent.piececount
            if agent.piececount > max_piececount:
                max_piececount = agent.piececount

            # 记录每个 step 的数据
            for step_data in trajectory:
                all_states.append(step_data["state"])
                all_ref_probs.append(step_data["ref_prob"])
                all_actions.append(step_data["action"])
                all_masks.append(1)  # all steps valid
                # 优势用最终奖励填充，后续在训练时标准化
                all_advantages.append(float(total_reward))

        print(f"\nCollected {len(all_states)} steps from {G} games")

        # GRPO 优势标准化：按组内标准化
        steps_per_game = len(all_states) // G
        if steps_per_game == 0:
            print("no data collected, return")
            return

        normalized_advantages = []
        for g in range(G):
            start = g * steps_per_game
            end = (g + 1) * steps_per_game if g < G - 1 else len(all_states)
            group_advs = all_advantages[start:end]
            if len(group_advs) == 0:
                continue
            group_mean = np.mean(group_advs)
            group_std = np.std(group_advs) + 1e-6
            for adv in group_advs:
                normalized_adv = (adv - group_mean) / group_std
                normalized_advantages.append(float(normalized_adv))

        all_advantages = normalized_advantages

        # 打印优势分布
        adv_array = np.array(all_advantages)
        print(f"Advantage stats: min={adv_array.min():.3f} mean={adv_array.mean():.3f} "
              f"max={adv_array.max():.3f} std={adv_array.std():.3f}")

        # 保存训练数据
        print("GRPO Self Play end. length: %s saving ..." % len(all_states))
        filetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        equi_data = get_equi_data(all_states, all_ref_probs, all_advantages, all_actions, all_masks)

        for i, obj in enumerate(equi_data):
            filename = f"{filetime}-{i}.pkl"
            savefile = os.path.join(data_wait_dir, filename)
            with open(savefile, "wb") as fn:
                pickle.dump(obj, fn)

        print(f"saved file basename: {filetime} length: {len(equi_data)}")

        # 更新训练状态
        state = read_status_file()
        state["total"]["agent"] += 1
        # 评估结果（test_play）
        state["total"]["max_score_grpo"] = max(state["total"]["max_score_grpo"], max_removedlines)
        state["total"]["max_piececount_grpo"] = max(state["total"]["max_piececount_grpo"], max_pieces_count)
        state["total"]["min_score_grpo"] = min(state["total"]["min_score_grpo"], min_removedlines)
        state["total"]["min_piececount_grpo"] = min(state["total"]["min_piececount_grpo"], min_pieces_count)
        # GRPO 游戏结果
        state["total"]["grpo_score"] = round(total_score / G, 3)
        state["total"]["grpo_piececount"] = round(total_piececount / G, 3)
        state["total"]["grpo_steps"] = round(total_steps / G, 3)
        state["total"]["grpo_min_piececount"] = min(state["total"]["grpo_min_piececount"], min_piececount)
        state["total"]["grpo_max_piececount"] = max(state["total"]["grpo_max_piececount"], max_piececount)
        save_status_file(state)

        print(f"saved file basename: {filetime} length: {len(equi_data)}")
        print(f"status updated: agent={state['total']['agent']}")

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
