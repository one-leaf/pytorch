import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from datetime import datetime
from transformer import GameTransformer

# 定义游戏的保存文件名和路径
model_name = ""
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, 'data', model_name)
if not os.path.exists(data_dir): os.makedirs(data_dir)
data_wait_dir = os.path.join(curr_dir, 'data', model_name, 'wait')
if not os.path.exists(data_wait_dir): os.makedirs(data_wait_dir)
model_dir = os.path.join(curr_dir, 'model', model_name)
if not os.path.exists(model_dir): os.makedirs(model_dir)
model_file =  os.path.join(model_dir, 'model.pth')
nan_log_file = os.path.join(model_dir, 'nan_log.txt')


def log_nan(msg):
    with open(nan_log_file, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


class PolicyNet():
    def __init__(self, input_width, input_height, output_size, model_file=None, device=None, l2_const=5e-5):
        self.input_channels = 2  # 输入通道数
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_width * input_height
        self.output_size = output_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        print("use", device)

        self.l2_const = l2_const
        self.net = GameTransformer(embed_dim=64, depth=2, num_heads=4,
                                                 num_actions=output_size, in_channels=2)
        self.net.to(device)

        self.optimizer = optim.AdamW(self.net.parameters(), lr=1e-5, weight_decay=self.l2_const)

        if model_file and os.path.exists(model_file):
            print("Loading model", model_file)
            net_sd = torch.load(model_file, map_location=self.device)
            if isinstance(net_sd, dict) and 'model_state_dict' in net_sd:
                self.net.load_state_dict(net_sd['model_state_dict'], strict=False)
            else:
                self.net.load_state_dict(net_sd, strict=False)
            print("Load weight success")
        else:
            print("Initializing new model", model_file)
            self.net.init_weights()
            self.save_model(model_file)
        self.lr = 0

    def print_network(self):
        x = torch.Tensor(1,2,20,10).to(self.device)
        prev_action = torch.LongTensor([0]).to(self.device)
        print(self.net)
        log_probs, value = self.net(x, prev_action)
        print("log_probs:", log_probs.size(), "value:", value.size())
        print("policy probs:", torch.exp(log_probs).size())

    def policy(self, state_batch, prev_action):
        """
        输入: 一组游戏的当前状态 [B, 2, 20, 10], 上一步动作 [B]
        输出: 一组动作的概率
        """
        if torch.is_tensor(state_batch):
            state_batch_tensor = state_batch.to(self.device)
        else:
            state_batch_tensor = torch.FloatTensor(state_batch).to(self.device)

        if not torch.is_tensor(prev_action):
            prev_action = torch.LongTensor(prev_action).to(self.device)
        else:
            prev_action = prev_action.to(self.device)

        self.net.eval()
        with torch.no_grad():
            act_probs, _ = self.net.forward(state_batch_tensor, prev_action)

        act_probs = np.exp(act_probs.cpu().numpy())
        return act_probs
        

    # PPO 训练步骤（带 Value Head + GAE 信用分配 + 分位数价值）
    def train_step_ppo(self, state_batch, log_probs_old, action_batch, prev_action_batch,
                        game_ids, R_batch, is_terminal_batch, G_batch, lr,
                        clip_eps=0.2, beta=0.05, entropy_weight=0.01,
                        gamma=0.99, lam=0.95, vf_coef=0.5, availables_batch=None):
        """PPO + V(s) 训练步骤（分位数价值头 + 步重要性加权 + availables mask）
        - V(s): value head 估计每步状态价值
        - GAE: 步级别信用分配（替代线性衰减）
        - policy_loss: PPO clip 损失
        - value_loss: MSE 损失
        - kl_loss: KL 散度惩罚
        - entropy: 熵正则化（仅计算有效动作）
        - availables: 有效动作 mask，无效动作概率强制为 0
        """
        # 每次更新学习率（lr_multiplier 动态调整）
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        B = state_batch.shape[0]

        # availables mask: [B, num_actions]，1=有效，0=无效
        if availables_batch is not None:
            availables_t = torch.FloatTensor(availables_batch).to(self.device)
        else:
            availables_t = torch.ones(B, self.output_size, device=self.device)

        log_probs_old_t = torch.FloatTensor(log_probs_old).to(self.device)
        # 对 log_probs_old 也应用 mask + clamp + renorm，和当前策略处理方式一致
        # probs_old = torch.exp(log_probs_old_t)
        # probs_old = probs_old * availables_t
        # probs_old = probs_old.clamp(min=0.02, max=0.98) * availables_t
        # probs_old_sum = probs_old.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # probs_old = probs_old / probs_old_sum
        # log_probs_old_t = torch.log(probs_old)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        prev_action_batch = torch.LongTensor(prev_action_batch).to(self.device)
        game_ids = game_ids.tolist() if hasattr(game_ids, 'tolist') else list(game_ids)

        # ── 参数校验 ─────────────────────────────────────────────
        assert state_batch.shape == (B, 2, 20, 10), f"state shape mismatch: {state_batch.shape}"
        assert not torch.isnan(state_batch).any(), "state_batch contains NaN"
        assert action_batch.min() >= 0 and action_batch.max() < self.output_size, \
            f"action out of [0,{self.output_size}): [{action_batch.min()},{action_batch.max()}]"
        assert prev_action_batch.min() >= 0 and prev_action_batch.max() < self.output_size, \
            f"prev_action out of [0,{self.output_size}): [{prev_action_batch.min()},{prev_action_batch.max()}]"
        assert log_probs_old_t.shape == (B, self.output_size), \
            f"log_probs_old shape mismatch: {log_probs_old_t.shape}"
        # game_ids 同一游戏的步骤必须在 batch 中连续（shuffle=False 保证）
        seen = set()
        prev_gid = None
        for gid in game_ids:
            if gid != prev_gid:
                assert gid not in seen, \
                    f"game_id '{gid}' appears non-contiguously in batch — DataLoader must use shuffle=False"
                seen.add(gid)
                prev_gid = gid

        R_batch = torch.FloatTensor(R_batch).to(self.device)
        is_terminal_batch = torch.FloatTensor(is_terminal_batch).to(self.device)
        G_batch = torch.FloatTensor(G_batch).to(self.device)

        self.net.train()
        log_probs, values = self.net(state_batch, prev_action_batch)

        # 概率处理：先 mask 无效动作，再 clamp 有效动作，最后 renorm
        # probs = torch.exp(log_probs)
        # probs = probs * availables_t                          # 无效动作概率归零
        # valid_probs = probs.clamp(min=0.02, max=0.98)         # 有效动作 clamp 到 [2%, 98%]
        # valid_probs = valid_probs * availables_t              # 再次确保无效动作为 0
        # probs = valid_probs / valid_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # log_probs = torch.log(probs)
        # values: [B, N] quantiles

        # 中间 1/2 分位数均值作为标量 V(s) 用于 GAE（trimmed mean，抗极端分位数扰动）
        N_q = values.shape[1]
        lo = N_q // 4          # index 2
        hi = N_q - N_q // 4    # index 6
        v_scalar = values[:, lo:hi].mean(dim=1)  # [B]
        v_scalar = torch.clamp(v_scalar, -10.0, 10.0)

        # 分位数 spread：分位数方差，衡量 V(s) 估计的不确定性
        # 方差越大 → 该步价值越不确定 → 越值得投入梯度
        taus = (torch.arange(N_q, device=self.device) + 0.5) / N_q  # [N]
        spread = ((values - v_scalar.unsqueeze(1)) ** 2).sum(-1)   # [B]  Σ(Q(τ)-median)²
        spread = spread.clamp(min=1e-4)  # 防止完全确定时除零

        # ── GAE: 按游戏分组计算步级别优势 ──────────
        B = v_scalar.shape[0]
        advantages = torch.zeros(B, device=self.device)  # GAE advantage

        for gid in set(game_ids):
            idx = [i for i, g in enumerate(game_ids) if g == gid]
            n = len(idx)

            V = v_scalar[idx]

            # GAE advantage: A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
            # δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
            rewards = R_batch[idx]  # 使用原始奖励，和 V(s) 尺度一致
            V_next = torch.zeros(n, device=self.device)
            V_next[:-1] = V[1:].detach()
            # 最后一步：游戏结束 → V_next=0；batch截断 → bootstrap V(s)
            last_terminal = is_terminal_batch[idx[-1]]
            V_next[-1] = V[-1].detach() * (1 - last_terminal)
            deltas = rewards + gamma * V_next - V.detach()

            gae = torch.zeros(n, device=self.device)
            gae[-1] = deltas[-1]
            for t in range(n - 2, -1, -1):
                gae[t] = deltas[t] + gamma * lam * gae[t + 1]
            advantages[idx] = gae

            # 调试：打印第一局的详细信息
            # if gid == list(set(game_ids))[0]:
            #     print(f"\n=== Game {gid} ({n} steps) ===")
            #     print(f"r_t:     {rewards.cpu().numpy()}")
            #     print(f"G_batch:{G_batch[idx].cpu().numpy()}")
            #     print(f"V(s):    {V.detach().cpu().numpy()}")
            #     print(f"adv:     {gae.cpu().numpy()}")

        # 全局标准化 advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-3)
        advantages = (advantages - adv_mean) / adv_std

        # ── Policy loss (PPO clip + 步重要性加权) ─────────────────
        actions = action_batch.unsqueeze(-1)
        log_prob_new = log_probs.gather(-1, actions)                        # [B, 1]
        log_prob_old = log_probs_old_t.gather(-1, actions).squeeze(-1)      # [B]

        log_ratio = torch.clamp(log_prob_new.squeeze(-1) - log_prob_old, -10.0, 10.0)
        ratios = torch.exp(log_ratio)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages

        # 重要性加权：spread 大的步骤获得更大梯度
        step_weight = spread / spread.mean().detach()
        step_weight = (0.5 + step_weight.clamp(max=1.5)).detach()  # [B]
        policy_loss = -(torch.min(surr1, surr2) * step_weight).mean()

        # ── Value loss (Quantile Huber: 分位数回归) ───────────────
        target_exp = G_batch.unsqueeze(1).expand_as(values)  # [B, N]
        diff = values - target_exp
        q_weights = torch.where(diff > 0, taus.unsqueeze(0), 1 - taus.unsqueeze(0))
        value_loss = (q_weights * F.smooth_l1_loss(values, target_exp, reduction='none')).mean()

        # ── KL 散度 ──────────────────────────────────────────────
        # 0.01 策略几乎没变，训练非常保守
        # 0.05~0.15 健康范围，策略在稳步更新
        # 0.2~0.3 变化较大，可能需要降低学习率
        # 0.5+  策略偏移严重，训练不稳定
        log_probs_safe = torch.clamp(log_probs, min=-20.0)
        probs_new = torch.exp(log_probs_safe)
        # KL 仅在有效动作上计算（无效动作 prob=0，不贡献 KL）
        kl_div = (probs_new * (log_probs_safe - log_probs_old_t) * availables_t).sum(dim=-1).mean()

        # ── 熵正则化 ─────────────────────────────────────────────
        # entropy_weight=1.0: 强制保持探索防止策略崩溃
        # 仅计算有效动作的 entropy，无效动作 prob=0 不参与
        # 有效动作数影响最大 entropy：N 个有效动作 → max entropy = log(N)
        entropy = -(probs_new * log_probs_safe * availables_t).sum(dim=-1).mean()

        # ── NONE 概率正则化：防止 NONE 被动累积 ──────────────────
        none_penalty_coef = 0.02
        none_penalty = probs_new[:, 3].mean()

        # ── 总损失 ───────────────────────────────────────────────
        loss = policy_loss + vf_coef * value_loss + beta * kl_div - entropy_weight * entropy + none_penalty_coef * none_penalty

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

        has_nan_grad = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in self.net.parameters() if p.grad is not None
        )
        if has_nan_grad:
            nan_params = [name for name, p in self.net.named_parameters()
                          if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())]
            msg = (f"GRAD NaN | policy_loss={policy_loss.item():.6f} value_loss={value_loss.item():.6f} "
                   f"kl_div={kl_div.item():.6f} entropy={entropy.item():.6f} loss={loss.item():.6f} | "
                   f"v_scalar=[{v_scalar.min().item():.4f}, {v_scalar.max().item():.4f}] "
                   f"spread=[{spread.min().item():.4f}, {spread.max().item():.4f}] "
                   f"adv=[{advantages.min().item():.4f}, {advantages.max().item():.4f}] | "
                   f"nan_params={nan_params[:10]}")
            print(f"\n[NaN GRAD] {msg}")
            log_nan(msg)
            self.optimizer.zero_grad()
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(float('nan'))

        self.optimizer.step()

        # 指标
        predicted = torch.argmax(log_probs, dim=1)
        accuracy = (predicted == action_batch).float().mean()

        return accuracy.item(), kl_div.item(), entropy.item(), value_loss.item()

    # 保存模型
    def save_model(self, model_file):
        """ save model params to file """
        torch.save(self.net.state_dict(), model_file)