'''
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/vision_transformer/vit_model.py
'''
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class PatchEmbed(nn.Module):
    """
    图片转嵌入数据，由 [B, C, H, W] -> [B, HW, C]
    """
    def __init__(self, img_size=(224,244), patch_size=(16,16), in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.img_size = (image_height, image_width)
        self.patch_size = (patch_height, patch_width)
        self.num_patches =  (image_height // patch_height) * (image_width // patch_width)

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 确保softmax前的方差为1
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # 计算 q,k,v 的多头转移矩阵
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 分割多头 query key value
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 通过q和k计算多头注意力权重
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 将多头注意力权重和多头v相乘得到v的多头加权值，然后合并成一个特征图
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
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

# 注意这里的drop的算法，保持了总体合计值
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
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
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

# 每一层
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        # nn.init.trunc_normal_(m.weight, std=.01)
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

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=(16,16), in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (tuple): 输入图片的尺寸
            patch_size (tuple): patch 尺寸，需要整除 img_size
            in_c (int): 输入图片的层数
            num_classes (int): 分类的种类 如果为-1或0 则不采用分类
            embed_dim (int): embedding 的维度
            depth (int): transformer 的深度
            num_heads (int): 注意力的个数
            mlp_ratio (int): mlp 隐藏层 维度与 embedding 的维度的比
            qkv_bias (bool): qkv 是否使用 bias
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): 是否在 DeiT models 中包含 蒸馏的 token 和 head
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): 数据归一化层
            act_layer:  (nn.Module): 激活层
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # 特征值保持一致
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 图片转换为 patch embedding [B, C, H, W] ==> [B, num_patches, embed_dim] 
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # 图片分割后的块数
        num_patches = self.patch_embed.num_patches                      # 196

        # 分类层
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))     # [1, 1, 768]
        # 蒸馏层
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None # [1, 1, 768]
        # 位置层
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # [1, 196+(1|2), 768]
        # 位置层的损失函数
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 深度 Dropout 衰减规则
        # 从0到最高drop比例，按深度递增，跨度是均分，得到一个深度衰减比例
        # 这样前面层的衰减低，后面层的衰减高
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  

        # 按照深度创建每一层的模块
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 表现层，如果定义了表现层尺寸且没有定义蒸馏层
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # 分类头、蒸馏头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()     # [B, 768] => [B, 1000]
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()  # [B, 768] => [B, 1000]

        # 参数初始化
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            # nn.init.trunc_normal_(self.dist_token, std=0.02)
            nn.init.normal_(self.dist_token, std=0.02)

        # nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768] 这里每一个B的 cls_token 都是一样的，并没有复制 cls_token 到每一个B
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 如果有至少蒸馏层也加上 
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)    # [B, 198, 768]

        # x 加上位置层，并且Dropout
        x = self.pos_drop(x + self.pos_embed)       # [B, 198, 768]

        # x 到达每一层的模块，包含了按照深度，由小到大的Dropout
        x = self.blocks(x)                    # [B, 198, 768]

        # 归一化
        x = self.norm(x)

        # 如果没有蒸馏层，则返回第一层分类层（加了fc+tanh），否则返回第一层和第二层蒸馏层的输出
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])         # [B, 768]
        else:
            return x[:, 0], x[:, 1]                 # [B, 768], [B, 768]         

    def forward(self, x):
        # 特征提取
        x = self.forward_features(x)
        # 分类头
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])       # [B, 1000], [B, 1000]
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2             # [B, 1000]
        else:
            x = self.head(x)                         # [B, 1000]
        return x


class VitNet(nn.Module):
    def __init__(self,  embed_dim=768, drop_ratio=0.1, drop_path_ratio=0.1, depth=12, num_heads=12, 
                        mlp_ratio=4.0, qkv_bias=True, qk_scale=None, attn_drop_ratio=0.3, num_classes=1000, num_quantiles=12):
        super(VitNet, self).__init__()

        assert embed_dim % num_heads==0, "embed_dim must be divisible by num_heads"

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer =  nn.GELU
        # 图片转换为 patch embedding [B, C, H, W] ==> [B, num_patches, embed_dim] 
        self.patch_embed = PatchEmbed(img_size=(20,10), patch_size=(1,5), in_c=3, embed_dim=embed_dim)
        # 图片分割后的块数
        num_patches = self.patch_embed.num_patches                      # p

        # 价值
        self.val_token = nn.Parameter(torch.zeros(1, 1, embed_dim))    # [1, 1, 768]
        # 位置层
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) # [1, p+1, 768]
        
        # 输入损失
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 深度 Dropout 衰减规则
        # 从0到最高drop比例，按深度递增，跨度是均分，得到一个深度衰减比例
        # 这样前面层的衰减低，后面层的衰减高
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  

        # 按照深度创建每一层的模块
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.act_fc = nn.Linear(embed_dim, embed_dim)  # [B, 768] => [B, 768]
        self.act_fc_act = nn.GELU()
        self.act_dist = nn.Linear(embed_dim, num_classes)  # [B, 768] => [B, 5]
        # self.act_dist_act = nn.Softmax(dim=1)

        self.val_fc = nn.Linear(embed_dim, embed_dim)   # [B, 768] => [B, 768]
        self.val_fc_act = nn.GELU()
        self.val_dist = nn.Linear(embed_dim, num_quantiles)   # [B, 768] => [B, 1]
        # self.val_dist_act = nn.Tanh()

        # 参数初始化, 这里需要pytorch 1.6以上版本
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # nn.init.trunc_normal_(self.val_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.val_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        # 特征提取
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 50, 768]
        # [1, 1, 768] -> [B, 1, 768] 这里每一个B的 token 都是一样的，并没有复制 token 到每一个B
        val_token = self.val_token.expand(x.shape[0], -1, -1)
        x = torch.cat((val_token, x), dim=1)    # [B, p+1, 768]

        # x 加上位置层，并且Dropout
        x = self.pos_drop(x + self.pos_embed)       # [B, p+1, 768]

        # x 到达每一层的模块，包含了按照深度，由小到大的Dropout
        x = self.blocks(x)                    # [B, p+1, 768]

        # 归一化
        x = self.norm(x)                        # [B, p+1, 768]

        # act = x[:, 1:].mean(dim = 1)             # [B, 768]
        act = x.max(dim = 1).values                 # [B, 768]
        act = self.act_fc(act)
        act = self.act_fc_act(act)
        act = self.act_dist(act)                # [B, num_classes]
        # act = self.act_dist_act(act)

        val = x[:, 0]                           # [B, 768]
        val = self.val_fc(val)
        val = self.val_fc_act(val)
        val = self.val_dist(val)                # [B, num_quantiles]
        # val = self.val_dist_act(val)

        return act, val        