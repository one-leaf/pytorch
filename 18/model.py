import torch
import torch.optim as optim
import os
import numpy as np
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

        self.r_running_mean = 15.0
        self.r_running_std = 1.0
        self.r_ema_alpha = 0.01

        if model_file and os.path.exists(model_file):
            print("Loading model", model_file)
            checkpoint = torch.load(model_file, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.r_running_mean = checkpoint.get('r_running_mean', 15.0)
                self.r_running_std = checkpoint.get('r_running_std', 1.0)
                print(f"Load model + running stats (mean={self.r_running_mean:.2f}, std={self.r_running_std:.2f})")
            else:
                self.net.load_state_dict(checkpoint, strict=False)
                print("Load model (old format, no running stats)")
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
        

    # GRPO 训练步骤（带 Value Head + GAE 信用分配）
    def train_step_grpo(self, state_batch, ref_probs, log_probs_old, action_batch, _mask_batch, prev_action_batch,
                        game_ids, R_batch, lr,
                        clip_eps=0.2, beta=0.05, entropy_weight=0.01,
                        gamma=0.99, lam=0.95, vf_coef=0.5):
        """GRPO + V(s) 训练步骤
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
        log_probs_old_t = torch.FloatTensor(log_probs_old).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        prev_action_batch = torch.LongTensor(prev_action_batch).to(self.device)
        game_ids = game_ids.tolist() if hasattr(game_ids, 'tolist') else list(game_ids)
        R_batch = torch.FloatTensor(R_batch).to(self.device)

        self.net.train()
        log_probs, values = self.net(state_batch, prev_action_batch)
        values = values.squeeze(-1)  # [B]

        # ── 用 EMA running stats 归一化 R ────────────────────────────
        r_mean = R_batch.mean().item()
        r_std = R_batch.std().item() + 1e-8
        self.r_running_mean = self.r_ema_alpha * self.r_running_mean + (1 - self.r_ema_alpha) * r_mean
        self.r_running_std = self.r_ema_alpha * self.r_running_std + (1 - self.r_ema_alpha) * r_std
        R_norm = (R_batch - self.r_running_mean) / (self.r_running_std + 1e-8)
        R_norm = torch.clamp(R_norm, -5.0, 5.0)

        # ── GAE: 按游戏分组计算步级别优势 ──────────────────────────
        B = values.shape[0]
        advantages = torch.zeros(B, device=self.device)

        for gid in set(game_ids):
            idx = [i for i, g in enumerate(game_ids) if g == gid]
            if len(idx) <= 1:
                advantages[idx[0]] = R_norm[idx[0]] - values[idx[0]].detach()
                continue

            V = values[idx]
            R = R_norm[idx[0]]
            n = len(idx)

            # rewards: 全零，最后一步为 R_norm
            rewards = torch.zeros(n, device=self.device)
            rewards[-1] = R

            # V_next: 每步的下一步价值，最后一步为 0（游戏结束）
            V_next = torch.zeros(n, device=self.device)
            V_next[:-1] = V[1:].detach()

            # TD error
            deltas = rewards + gamma * V_next - V.detach()

            # GAE: A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
            gae = torch.zeros(n, device=self.device)
            gae[-1] = deltas[-1]
            for t in range(n - 2, -1, -1):
                gae[t] = deltas[t] + gamma * lam * gae[t + 1]
            advantages[idx] = gae

        # 全局标准化
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ── Policy loss (PPO clip) ────────────────────────────────
        actions = action_batch.unsqueeze(-1)
        log_prob_new = log_probs.gather(-1, actions)                        # [B, 1]
        log_prob_old = log_probs_old_t.gather(-1, actions).squeeze(-1)      # [B]

        log_ratio = torch.clamp(log_prob_new.squeeze(-1) - log_prob_old, -10.0, 10.0)
        ratios = torch.exp(log_ratio)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value loss (MSE: V(s) 预测归一化后的 R) ─────────────────
        value_loss = ((values - R_norm) ** 2).mean()

        # ── KL 散度 ──────────────────────────────────────────────
        log_probs_safe = torch.clamp(log_probs, min=-20.0)
        probs_new = torch.exp(log_probs_safe)
        kl_div = (probs_new * (log_probs_safe - ref_log_probs)).sum(dim=-1).mean()

        # ── 熵正则化 ─────────────────────────────────────────────
        entropy = -(probs_new * log_probs_safe).sum(dim=-1).mean()

        # ── 总损失 ───────────────────────────────────────────────
        loss = policy_loss + vf_coef * value_loss + beta * kl_div - entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 指标
        predicted = torch.argmax(log_probs, dim=1)
        accuracy = (predicted == action_batch).float().mean()

        return accuracy.item(), kl_div.item(), entropy.item(), value_loss.item(), self.r_running_mean, self.r_running_std

    # 保存模型
    def save_model(self, model_file):
        """ save model params + running stats to file """
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'r_running_mean': self.r_running_mean,
            'r_running_std': self.r_running_std,
        }, model_file)