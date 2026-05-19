'''
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/vision_transformer/vit_model.py
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
'''
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ParallelThingsBlock(nn.Module):
    """ Parallel ViT block (N parallel attention followed by N parallel MLP)
    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    def __init__(
            self,
            dim,
            num_heads,
            num_parallel=2,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=False,
            drop_ratio=0.,
            attn_drop_ratio=0.,
            drop_path_ratio=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio
                )),
                ('drop_path', DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(
                    in_features=dim, hidden_features=int(dim * mlp_ratio),
                    act_layer=act_layer, drop=drop_ratio,
                )),
                ('drop_path', DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity())
            ])))

    def forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class VitPatchEmbed(nn.Module):
    """
    图片转嵌入数据，由 [B, C, H, W] -> [B, HW, C]
    """
    def __init__(self, img_size=(20,10), in_c=4, kernel_size=(5,5), embed_dim=768, padding=0, stride=(5,5), norm_layer=None, drop_ratio=0):
        super().__init__()
        image_height, image_width = pair(img_size)
        kernel_height, kernel_width = pair(kernel_size)
        padding_height, padding_width = pair(padding)
        stride_height, stride_width = pair(stride)
        self.dropout = nn.Dropout(p=drop_ratio)
        self.gelu = nn.GELU()
        self.proj_init = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj2 = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3 = nn.Conv2d(in_c, in_c, kernel_size=5, stride=1, padding=2, bias=False)
        self.proj4 = nn.Conv2d(in_c, in_c, kernel_size=7, stride=1, padding=3, bias=False)
        self.proj_end = nn.Conv2d(in_c*5, in_c*5, kernel_size=1, stride=1, padding=0, bias=False)
        self.out = nn.Conv2d(in_c*5, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.num_patches = ((image_height-kernel_height+2*padding_height)//stride_height+1) * ((image_width-kernel_width+2*padding_width)//stride_width+1)
        print("sequence length:",self.num_patches)

    def forward(self, x):
        x = self.proj_init(x)
        x = self.dropout(x)

        x1 = self.gelu(self.proj1(x))

        x2 = self.gelu(self.proj2(x))

        x3 = self.gelu(self.proj3(x))

        x4 = self.gelu(self.proj4(x))

        x = torch.cat((x, x1, x2, x3, x4), dim=1)

        x = self.gelu(self.proj_end(x))
        x = self.dropout(x)

        x = self.out(x)
        x = self.dropout(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class VitNet(nn.Module):
    def __init__(self, embed_dim=768, drop_ratio=0.1, drop_path_ratio=0.1, depth=12, num_heads=12,
                        mlp_ratio=4.0, qkv_bias=True, qk_scale=None, attn_drop_ratio=0.3, num_classes=1000, num_quantiles=64, num_channels=4):
        super(VitNet, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        self.patch_embed = VitPatchEmbed(img_size=(20,10), in_c=num_channels, kernel_size=(2,2), stride=(2,2), padding=(0,0), embed_dim=embed_dim, drop_ratio=drop_ratio)
        num_patches = self.patch_embed.num_patches

        self.act_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.val_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]

        self.blocks = nn.Sequential(*[
            ParallelThingsBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.act_dist = nn.Linear(embed_dim, num_classes)
        self.act_dist_act = nn.LogSoftmax(dim=1)

        self.val_dist = nn.Linear(embed_dim, num_quantiles)

    def init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.act_token, std=0.02)
        nn.init.normal_(self.val_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        act_token = self.act_token.expand(x.shape[0], -1, -1)
        val_token = self.val_token.expand(x.shape[0], -1, -1)

        x = torch.cat((act_token, x, val_token), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)

        x = self.norm(x)
        act = x[:, 0]
        act = self.act_dist(act)
        act = self.act_dist_act(act)

        val = x[:, -1]
        val = self.val_dist(val)

        return act, val
