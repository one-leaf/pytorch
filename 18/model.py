import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from datetime import datetime
from transformer import GameTransformer

# 定义游戏的保存文件名和路径
model_name = "vit-ti" # "vit" # "mlp"
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
    def train_step_ppo(self, state_batch, ref_probs, log_probs_old, action_batch, _mask_batch, prev_action_batch,
                        game_ids, R_batch, lr, r_mean, r_std,
                        clip_eps=0.2, beta=0.05, entropy_weight=0.01,
                        gamma=0.99, lam=0.95, vf_coef=0.5):
        """PPO + V(s) 训练步骤（分位数价值头 + 步重要性加权）
        - V(s): value head 估计每步状态价值
        - GAE: 步级别信用分配（替代线性衰减）
        - policy_loss: PPO clip 损失
        - value_loss: MSE 损失
        - kl_loss: KL 散度惩罚
        - entropy: 熵正则化
        """
        # 每次更新学习率（lr_multiplier 动态调整）
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        ref_log_probs = torch.log(torch.FloatTensor(ref_probs) + 1e-10).to(self.device)
        ref_log_probs = torch.clamp(ref_log_probs, min=-20.0)
        log_probs_old_t = torch.FloatTensor(log_probs_old).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        prev_action_batch = torch.LongTensor(prev_action_batch).to(self.device)
        game_ids = game_ids.tolist() if hasattr(game_ids, 'tolist') else list(game_ids)
        R_batch = torch.FloatTensor(R_batch).to(self.device)

        self.net.train()
        log_probs, values = self.net(state_batch, prev_action_batch)
        # values: [B, N] quantiles

        # 中位数作为标量 V(s) 用于 GAE
        N_q = values.shape[1]
        v_scalar = values[:, N_q // 2]  # [B] median
        v_scalar = torch.clamp(v_scalar, -10.0, 10.0)

        # 分位数 spread：条件方差的代理，衡量步重要性
        taus = (torch.arange(N_q, device=self.device) + 0.5) / N_q  # [N]
        spread = (values * taus).sum(-1) - (values * (1 - taus)).sum(-1)  # [B]
        spread = spread.clamp(min=0.01)

        # ── 逐步奖励归一化 ──────────────────────────────────────────
        # R_batch 现在是每步即时奖励（方块落地=1, 其他=0）
        r_step = (R_batch - r_mean) / (r_std + 1e-3)
        r_step = torch.clamp(r_step, -5.0, 5.0)

        # ── GAE: 按游戏分组计算步级别优势 + value target ──────────
        B = v_scalar.shape[0]
        advantages = torch.zeros(B, device=self.device)
        v_targets = torch.zeros(B, device=self.device)

        for gid in set(game_ids):
            idx = [i for i, g in enumerate(game_ids) if g == gid]
            n = len(idx)

            if n <= 1:
                # 单步：advantage = r - V(s), target = r
                advantages[idx[0]] = r_step[idx[0]] - v_scalar[idx[0]].detach()
                v_targets[idx[0]] = r_step[idx[0]]
                continue

            V = v_scalar[idx]
            rewards = r_step[idx]

            # 折扣回报：G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
            returns = torch.zeros(n, device=self.device)
            returns[-1] = rewards[-1]
            for t in range(n - 2, -1, -1):
                returns[t] = rewards[t] + gamma * returns[t + 1]
            v_targets[idx] = returns

            # GAE advantage: A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
            # δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
            V_next = torch.zeros(n, device=self.device)
            V_next[:-1] = V[1:].detach()
            deltas = rewards + gamma * V_next - V.detach()

            gae = torch.zeros(n, device=self.device)
            gae[-1] = deltas[-1]
            for t in range(n - 2, -1, -1):
                gae[t] = deltas[t] + gamma * lam * gae[t + 1]
            advantages[idx] = gae

            # 调试：打印第一局的详细信息
            if gid == list(set(game_ids))[0]:
                print(f"\n=== Game {gid} ({n} steps) ===")
                print(f"r_t:     {rewards.cpu().numpy()}")
                print(f"v_target:{returns.cpu().numpy()}")
                print(f"V(s):    {V.detach().cpu().numpy()}")
                print(f"adv:     {gae.cpu().numpy()}")

        # G 统计（折扣回报 v_targets）
        g_mean = v_targets.mean().item()
        g_std = v_targets.std().item()

        # 全局标准化
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
        # 步重要性加权：spread 大的步骤获得更大梯度
        step_weight = spread / spread.mean().detach()
        step_weight = (0.5 + step_weight.clamp(max=1.5)).detach()  # [B]
        policy_loss = -(torch.min(surr1, surr2) * step_weight).mean()

        # ── Value loss (Quantile Huber: 分位数回归) ───────────────
        target_exp = v_targets.unsqueeze(1).expand_as(values)  # [B, N]
        diff = values - target_exp
        q_weights = torch.where(diff > 0, taus.unsqueeze(0), 1 - taus.unsqueeze(0))
        value_loss = (q_weights * F.smooth_l1_loss(values, target_exp, reduction='none')).mean()

        # ── KL 散度 ──────────────────────────────────────────────
        log_probs_safe = torch.clamp(log_probs, min=-20.0)
        probs_new = torch.exp(log_probs_safe)
        kl_div = (probs_new * (log_probs_safe - ref_log_probs)).sum(dim=-1).mean()

        # ── 熵正则化 ─────────────────────────────────────────────
        # 1.0 2-3 个动作在竞争
        # 0.8 让策略保持 2-3 个动作的竞争。这样既有堆叠能力，又有机会探索消行
        # 0.5 基本确定性，探索很弱
        # 0.3 接近完全确定性
        entropy = -(probs_new * log_probs_safe).sum(dim=-1).mean()

        # ── 总损失 ───────────────────────────────────────────────
        loss = policy_loss + vf_coef * value_loss + beta * kl_div - entropy_weight * entropy

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
                   f"r_step=[{r_step.min().item():.4f}, {r_step.max().item():.4f}] "
                   f"adv=[{advantages.min().item():.4f}, {advantages.max().item():.4f}] | "
                   f"nan_params={nan_params[:10]}")
            print(f"\n[NaN GRAD] {msg}")
            log_nan(msg)
            self.optimizer.zero_grad()
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(float('nan')), 0.0, 0.0

        self.optimizer.step()

        # 指标
        predicted = torch.argmax(log_probs, dim=1)
        accuracy = (predicted == action_batch).float().mean()

        return accuracy.item(), kl_div.item(), entropy.item(), value_loss.item(), g_mean, g_std

    # 保存模型
    def save_model(self, model_file):
        """ save model params to file """
        torch.save(self.net.state_dict(), model_file)