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
        

    # PPO 训练步骤（GAE advantage + 分位数价值）
    def train_step_ppo(self, state_batch, log_probs_old, action_batch, prev_action_batch,
                        gae_advantage_batch, td_target_batch, lr,
                        clip_eps=0.2, beta=0.05, entropy_weight=0.01,
                        vf_coef=0.5, availables_batch=None):
        """PPO + GAE 训练步骤（分位数价值头）
        - gae_advantage: 预计算的 GAE advantage（从后往前递推）
        - td_target: 预计算的 TD target（已限制范围 [-1, 0]）
        - policy_loss: PPO clip 损失，直接用 gae_advantage
        - value_loss: Quantile Huber 损失，回归 td_target
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

        # ── 参数校验 ─────────────────────────────────────────────
        assert state_batch.shape == (B, 2, 20, 10), f"state shape mismatch: {state_batch.shape}"
        assert not torch.isnan(state_batch).any(), "state_batch contains NaN"
        assert action_batch.min() >= 0 and action_batch.max() < self.output_size, \
            f"action out of [0,{self.output_size}): [{action_batch.min()},{action_batch.max()}]"
        assert prev_action_batch.min() >= 0 and prev_action_batch.max() < self.output_size, \
            f"prev_action out of [0,{self.output_size}): [{prev_action_batch.min()},{prev_action_batch.max()}]"
        assert log_probs_old_t.shape == (B, self.output_size), \
            f"log_probs_old shape mismatch: {log_probs_old_t.shape}"

        gae_advantages = torch.FloatTensor(gae_advantage_batch).to(self.device)
        gae_advantages = torch.nan_to_num(gae_advantages, nan=0.0)  # NaN 替换为 0

        td_target_tensor = torch.FloatTensor(td_target_batch).to(self.device)
        td_target_tensor = torch.clamp(td_target_tensor, -10.0, 10.0)  # 防止极端值

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

        # 中间 1/2 分位数均值作为标量 V(s)（trimmed mean，抗极端分位数扰动）
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

        # ── GAE advantage（离线预计算） ──────────────────────────
        # gae_advantages: 从 selfplay 的 v_t 从后往前递推计算，已包含多步回报信息
        # 优势：对稀疏奖励，能传播终端奖励信号到更多步
        advantages = gae_advantages  # [B] 直接使用预计算的 GAE

        # td_target: 离线预计算，直接用于 value loss 回归
        td_target = td_target_tensor  # [B]

        # 全局标准化 advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-3)
        if torch.isnan(adv_std) or torch.isinf(adv_std):
            adv_std = torch.tensor(1.0, device=self.device)  # std 为 NaN 时跳过标准化
        advantages = (advantages - adv_mean) / adv_std

        # ── Policy loss (PPO clip + 步重要性加权) ─────────────────
        # ratios = pi_new(a|s) / pi_old(a|s)
        # pi_old 是 selfplay 采样时的策略（与 v_next 来自同一版本模型）
        # ratios 修正新旧策略分布差异，使梯度方向对当前策略无偏
        # clip(ratios, 1-eps, 1+eps) 限制更新步长，防止策略跳变
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
        target_exp = td_target.unsqueeze(1).expand_as(values)  # [B, N]
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

        # ── 不可行动作概率惩罚：压低不可用动作的概率 ──────────────
        unavailable_penalty_coef = 1
        # 不可用位置的 log_prob，希望它们趋向 -inf（概率趋向 0）
        unavailable_log_probs = log_probs_safe * (1 - availables_t)  # [B, 5]，可用位置为 0
        # 取负值，希望 log_prob 越小越好；clamp 防止数值不稳定
        unavailable_penalty = torch.clamp(unavailable_log_probs.sum(dim=-1), min=-20.0).mean()

        # ── 总损失 ───────────────────────────────────────────────
        loss = policy_loss + vf_coef * value_loss + beta * kl_div - entropy_weight * entropy + none_penalty_coef * none_penalty + unavailable_penalty_coef * unavailable_penalty

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
            # 详细的 NaN 诊断信息
            msg = (f"GRAD NaN | policy_loss={policy_loss.item():.6f} value_loss={value_loss.item():.6f} "
                   f"kl_div={kl_div.item():.6f} entropy={entropy.item():.6f} loss={loss.item():.6f} | "
                   f"v_scalar=[{v_scalar.min().item():.4f}, {v_scalar.max().item():.4f}] "
                   f"spread=[{spread.min().item():.4f}, {spread.max().item():.4f}] | "
                   # 追踪 NaN 来源
                   f"td_target=[{td_target_tensor.min().item():.4f}, {td_target_tensor.max().item():.4f}, nan={torch.isnan(td_target_tensor).sum().item()}] "
                   f"gae=[{gae_advantages.min().item():.4f}, {gae_advantages.max().item():.4f}, nan={torch.isnan(gae_advantages).sum().item()}] "
                   f"adv_mean={adv_mean.item():.4f} adv_std={adv_std.item():.4f} "
                   f"adv=[{advantages.min().item():.4f}, {advantages.max().item():.4f}, nan={torch.isnan(advantages).sum().item()}] | "
                   f"log_prob_new=[{log_prob_new.min().item():.4f}, {log_prob_new.max().item():.4f}, nan={torch.isnan(log_prob_new).sum().item()}] "
                   f"log_prob_old=[{log_prob_old.min().item():.4f}, {log_prob_old.max().item():.4f}, nan={torch.isnan(log_prob_old).sum().item()}] "
                   f"ratios=[{ratios.min().item():.4f}, {ratios.max().item():.4f}, nan={torch.isnan(ratios).sum().item()}] | "
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