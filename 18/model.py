import torch
import torch.optim as optim
import os
import numpy as np
from vit import VitNet

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


class PolicyValueNet():
    def __init__(self, input_width, input_height, output_size, model_file=None, device=None, l2_const=5e-5):
        self.input_channels = 4  # 输入通道数
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_width * input_height
        self.output_size = output_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device=device
        print("use", device)

        self.l2_const = l2_const
        self.policy_value_net = VitNet(embed_dim=192, depth=6, num_heads=3, num_classes=5, num_quantiles=128, drop_ratio=0.1, drop_path_ratio=0.1, attn_drop_ratio=0.1, num_channels=self.input_channels)
        self.policy_value_net.to(device)

        self.optimizer = optim.AdamW(self.policy_value_net.parameters(), lr=1e-5, weight_decay=self.l2_const)

        if model_file and os.path.exists(model_file):
            print("Loading model", model_file)
            net_sd = torch.load(model_file, map_location=self.device)
            print("Load model", model_file, "success")
            print("Loading weight")
            self.policy_value_net.load_state_dict(net_sd, strict=False)
            print("Load weight success")
        else:
            print("Initializing new model", model_file)
            self.policy_value_net.init_weights()
            self.save_model(model_file)
        self.lr = 0

    def print_netwark(self):
        x = torch.Tensor(1,4,20,10).to(self.device)
        print(self.policy_value_net)
        v, p = self.policy_value_net(x)
        print("value:", v.size())
        print("policy:", p.size())

    def policy_value(self, state_batch):
        """
        输入: 一组游戏的当前状态
        输出: 一组动作的概率和动作的得分
        """
        if torch.is_tensor(state_batch):
            state_batch_tensor = state_batch.to(self.device)
        else:
            state_batch_tensor = torch.FloatTensor(state_batch).to(self.device)

        self.policy_value_net.eval()
        with torch.no_grad():
            act_probs, value = self.policy_value_net.forward(state_batch_tensor)

        value = torch.clamp(value, -1, 1)
        value = torch.mean(value, dim=1)

        act_probs = np.exp(act_probs.cpu().numpy())
        value = value.cpu().numpy()
        return act_probs, value
        

    # GRPO 训练步骤
    def train_step_grpo(self, state_batch, ref_probs, advantages, action_batch, mask_batch, lr,
                        clip_eps=0.2, beta=0.05, entropy_weight=0.01):
        """GRPO 训练步骤
        - policy_loss: PPO clip 损失，使用 GRPO 组相对优势
        - kl_loss: KL 散度惩罚，约束新策略相对参考策略
        - entropy: 熵正则化
        - 无 value loss（value head 保留但训练时不使用）
        """
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        ref_log_probs = torch.log(torch.FloatTensor(ref_probs) + 1e-10).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)

        self.policy_value_net.train()
        log_probs, values = self.policy_value_net(state_batch)
        # values 计算但不用于损失（纯 GRPO 不需要价值网络）

        # 取被选择动作的 log 概率
        actions = action_batch.unsqueeze(-1)
        log_prob_new = log_probs.gather(-1, actions)              # [B, 1]
        log_prob_old = ref_log_probs.gather(-1, actions).detach()  # [B, 1]

        # PPO 风格 ratio
        ratios = torch.exp(log_prob_new - log_prob_old)            # [B, 1]
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # KL 散度: sum(p_new * (log_p_new - log_p_ref))
        probs_new = torch.exp(log_probs)  # [B, 5]
        kl_div = (probs_new * (log_probs - ref_log_probs)).sum(dim=-1).mean()

        # 熵正则化
        entropy = -(probs_new * log_probs).sum(dim=-1).mean()

        # 总损失
        loss = policy_loss + beta * kl_div - entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 指标
        predicted = torch.argmax(log_probs, dim=1)
        accuracy = (predicted == action_batch).float().mean()

        return accuracy.item(), kl_div.item(), entropy.item()

    # 保存模型
    def save_model(self, model_file):
        """ save model params to file """
        torch.save(self.policy_value_net.state_dict(), model_file)