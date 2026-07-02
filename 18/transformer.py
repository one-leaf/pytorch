import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=2, embed_dim=64, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class GameTransformer(nn.Module):
    """Decoder-only transformer for game action prediction.
    Input:  state [B, 2, 20, 10], prev_action [B]
    Output: log_probs [B, num_actions]

    Sequence: [prev_action_token, 50 state patches, action_BOS_token]
    """
    def __init__(self, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.0,
                 patch_size=2, num_actions=5, in_channels=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_actions = num_actions

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        num_patches = (20 // patch_size) * (10 // patch_size)
        seq_len = 1 + num_patches + 1  # prev_action + patches + action_BOS

        self.prev_action_embed = nn.Embedding(num_actions, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.action_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.action_head = nn.Linear(embed_dim, num_actions)
        self.value_head = nn.Linear(embed_dim, 1)

        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GameTransformer params: {n_params:,}")
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
        x = self.decoder(tgt=x, memory=x, tgt_mask=causal_mask, memory_mask=causal_mask)

        action_logits = self.action_head(x[:, -1])
        value = self.value_head(x[:, -1])
        log_probs = torch.log_softmax(action_logits, dim=-1)
        return log_probs, value
