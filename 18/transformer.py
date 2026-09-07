import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=2, embed_dim=64, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MoELayer(nn.Module):
    """Mixture of Experts layer with load balancing loss"""
    def __init__(self, embed_dim, ffn_dim, num_experts=7, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 创建 num_experts 个 expert
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, embed_dim)
            )
            for _ in range(num_experts)
        ])

        # 门控网络
        self.router = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        """
        x: [B, D] 输入特征
        返回: (output, aux_loss)
        """
        # 计算路由权重 [B, num_experts]
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # 选择 top-k 个 expert
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # 归一化 top-k 权重
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 计算输出：加权合并各 expert 的结果
        output = torch.zeros_like(x)  # [B, D]

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # [B]
            expert_weights = top_k_probs[:, k:k+1]  # [B, 1]

            # 对每个 expert，找到路由到它的 token
            for e in range(self.num_experts):
                mask = (expert_indices == e)  # [B]
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weights[mask] * expert_output

        # Load balancing loss: 鼓励各 expert 被均匀使用
        # f_i = 每个 expert 被选中的比例
        # p_i = 每个 expert 的平均路由概率
        # aux_loss = num_experts * Σ(f_i * p_i)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices.view(-1), self.num_experts).float()  # [B*top_k, num_experts]
            f = expert_mask.mean(dim=0)  # [num_experts]

        p = router_probs.mean(dim=0)  # [num_experts]
        aux_loss = self.num_experts * (f * p).sum()

        return output, aux_loss


class MoEDecoderLayer(nn.Module):
    """Transformer decoder layer with MoE FFN"""
    def __init__(self, embed_dim, num_heads, ffn_dim, num_experts=7, top_k=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # MoE FFN (replaces traditional FFN)
        self.moe = MoELayer(embed_dim, ffn_dim, num_experts, top_k)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        x: [B, L, D] 目标序列
        memory: [B, L, D] 记忆序列（self-attention 时用）
        tgt_mask: [L, L] causal mask
        memory_mask: [L, L] 用于 cross-attention
        返回: (output [B, L, D], aux_loss)
        """
        # Self-attention + residual + norm
        residual = x
        x_sa, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(residual + self.dropout1(x_sa))

        # Cross-attention + residual + norm
        residual = x
        x_ca, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = self.norm2(residual + self.dropout2(x_ca))

        # MoE FFN + residual + norm
        # MoE 对序列的每个 token 独立计算
        B, L, D = x.shape
        x_flat = x.view(B * L, D)  # [B*L, D]
        moe_out, aux_loss = self.moe(x_flat)
        moe_out = moe_out.view(B, L, D)  # [B, L, D]
        x = self.norm3(x + self.dropout3(moe_out))

        return x, aux_loss


class GameTransformer(nn.Module):
    """Decoder-only transformer for game action prediction with MoE.
    Input:  state [B, 2, 20, 10], prev_action [B]
    Output: log_probs [B, num_actions], value [B, num_quantiles], aux_loss

    Sequence: [prev_action_token, 50 state patches, action_BOS_token]
    """
    def __init__(self, embed_dim=32, depth=2, num_heads=4, mlp_ratio=2.0,
                 patch_size=2, num_actions=5, in_channels=2, dropout=0.1,
                 num_experts=7, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_actions = num_actions
        self.depth = depth

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        num_patches = (20 // patch_size) * (10 // patch_size)
        seq_len = 1 + num_patches + 1  # prev_action + patches + action_BOS

        self.prev_action_embed = nn.Embedding(num_actions, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.action_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # MoE decoder layers (replaces nn.TransformerDecoder)
        ffn_dim = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([
            MoEDecoderLayer(embed_dim, num_heads, ffn_dim, num_experts, top_k, dropout)
            for _ in range(depth)
        ])

        self.action_head = nn.Linear(embed_dim, num_actions)
        self.num_quantiles = 4
        self.value_head = nn.Linear(embed_dim, self.num_quantiles)

        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GameTransformer params: {n_params:,}")
        print(f"MoE config: {num_experts} experts, top-{top_k}")
        print(f"Sequence length: {seq_len}")

    def init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.action_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x, prev_action):
        B = x.shape[0]

        prev_token = self.prev_action_embed(prev_action).unsqueeze(1)  # [B, 1, D]
        patches = self.patch_embed(x)                                   # [B, 50, D]
        action_token = self.action_token.expand(B, -1, -1)             # [B, 1, D]

        x = torch.cat([prev_token, patches, action_token], dim=1)      # [B, 52, D]
        x = self.pos_drop(x + self.pos_embed)

        causal_mask = self.causal_mask.to(x.device)

        # 通过所有 MoE decoder layers
        aux_loss_total = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x, memory=x, tgt_mask=causal_mask, memory_mask=causal_mask)
            aux_loss_total = aux_loss_total + aux_loss

        action_logits = self.action_head(x[:, -1])
        value = self.value_head(x[:, -1])
        log_probs = torch.log_softmax(action_logits, dim=-1)
        return log_probs, value, aux_loss_total
