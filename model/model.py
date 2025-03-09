import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    # ConvNeXt的基本块，包含深度可分离卷积、层归一化、点卷积和激活函数
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度可分离卷积
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 第一个点卷积（1x1卷积），使用线性层实现
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 第二个点卷积（1x1卷积），使用线性层实现
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x
    
class ConvNeXtBlock_v2(nn.Module):
    # ConvNeXt的基本块，包含深度可分离卷积、层归一化、点卷积和激活函数
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 深度可分离卷积
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 第一个点卷积（1x1卷积），使用线性层实现
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 第二个点卷积（1x1卷积），使用线性层实现
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    # 支持两种数据格式的LayerNorm：channels_last（默认）或channels_first
    # channels_last：(N, H, W, C)
    # channels_first：(N, C, H, W)
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DropPath(nn.Module):
    # Drop paths (Stochastic Depth) per sample，应用于残差块的主路径中
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    # Drop paths (Stochastic Depth) per sample，应用于残差块的主路径中
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适用于不同维度的张量，而不仅仅是2D卷积
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化
    output = x.div(keep_prob) * random_tensor
    return output

class Attention(nn.Module):
    # 自注意力机制模块
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 查询、键、值的线性变换
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 输出线性变换
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # 将输入转换为 (B, N, C)，其中 N = H * W
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # 使torchscript高兴（不能使用tensor作为tuple）

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力分数
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)  # 计算加权和
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # 将输出转换回 (B, C, H, W)
        return x
    
class Downsample(nn.Module):
    # 下采样模块
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
        )

    def forward(self, x):
        return self.downsample(x)

class Up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(Up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,  mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x
    
class U_NeXt_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4,),
            LayerNorm(64, eps=1e-6, data_format="channels_first"),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
        )
        self.downsample1 = Downsample(64, 128)
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
        )
        self.downsample2 = Downsample(128, 256)
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
        )
        self.downsample3 = Downsample(256, 512)
        self.stage4 = nn.Sequential(
            ConvNeXtBlock(512),
            ConvNeXtBlock(512),
            ConvNeXtBlock(512),
        )
        self.norm = nn.LayerNorm(512, eps=1e-6)
        self.upsample4 = Up_conv(512, 256)
        self.upsample3 = Up_conv(256, 128)
        self.upsample2 = Up_conv(128, 64)
        self.upsample1 = Up_conv(64, 32, scale_factor=4)
        self.out = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.upconv4 = conv_block(512, 256)
        self.upconv3 = conv_block(256, 128)
        self.upconv2 = conv_block(128, 64)
        self.upconv1 = conv_block(32, 32)

    
    def forward(self, x):
        e1 = self.stem(x)
        # print(f"e1: {e1.shape}")
        e1 = self.stage1(e1)
        # print(f"e1: {e1.shape}")
        e2 = self.downsample1(e1)
        # print(f"e2: {e2.shape}")
        e2 = self.stage2(e2)
        # print(f"e2: {e2.shape}")
        e3 = self.downsample2(e2)
        # print(f"e3: {e3.shape}")
        e3 = self.stage3(e3)
        # print(f"e3: {e3.shape}")
        e4 = self.downsample3(e3)
        # print(f"e4: {e4.shape}")
        e4 = self.stage4(e4)
        # print(f"e4: {e4.shape}")

        d4 = self.upsample4(e4)
        # print(f"d4: {d4.shape}")
        d4 = torch.cat([d4, e3], dim=1)
        # print(f"d4: {d4.shape}")
        d4 = self.upconv4(d4)
        # print(f"d4: {d4.shape}")

        d3 = self.upsample3(d4)
        # print(f"d3: {d3.shape}")
        d3 = torch.cat([d3, e2], dim=1)
        # print(f"d3: {d3.shape}")
        d3 = self.upconv3(d3)
        # print(f"d3: {d3.shape}")

        d2 = self.upsample2(d3)
        # print(f"d2: {d2.shape}")
        d2 = torch.cat([d2, e1], dim=1)
        # print(f"d2: {d2.shape}")
        d2 = self.upconv2(d2)
        # print(f"d2: {d2.shape}")

        d1 = self.upsample1(d2)
        # print(f"d1: {d1.shape}")
        d1 = self.upconv1(d1)
        # print(f"d1: {d1.shape}")
        out = self.out(d1)
        # print(f"out: {out.shape}")
        return out

class U_NeXt_v2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4,),
            LayerNorm(64, eps=1e-6, data_format="channels_first"),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
        )
        self.downsample1 = Downsample(64, 128)
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
        )
        self.downsample2 = Downsample(128, 256)
        self.stage3 = nn.Sequential(
            ConvNeXtBlock_v2(256),
            ConvNeXtBlock_v2(256),
        )
        self.downsample3 = Downsample(256, 512)
        self.stage4 = nn.Sequential(
            ConvNeXtBlock_v2(512),
            ConvNeXtBlock_v2(512),
        )
        self.norm = nn.LayerNorm(512, eps=1e-6)
        self.upsample4 = Up_conv(512, 256)
        self.upsample3 = Up_conv(256, 128)
        self.upsample2 = Up_conv(128, 64)
        self.upsample1 = Up_conv(64, 32, scale_factor=4)
        self.out = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.upconv4 = conv_block(512, 256)
        self.upconv3 = conv_block(256, 128)
        self.upconv2 = conv_block(128, 64)
        self.upconv1 = conv_block(32, 32)

    
    def forward(self, x):
        e1 = self.stem(x)
        # print(f"e1: {e1.shape}")
        e1 = self.stage1(e1)
        # print(f"e1: {e1.shape}")
        e2 = self.downsample1(e1)
        # print(f"e2: {e2.shape}")
        e2 = self.stage2(e2)
        # print(f"e2: {e2.shape}")
        e3 = self.downsample2(e2)
        # print(f"e3: {e3.shape}")
        e3 = self.stage3(e3)
        # print(f"e3: {e3.shape}")
        e4 = self.downsample3(e3)
        # print(f"e4: {e4.shape}")
        e4 = self.stage4(e4)
        # print(f"e4: {e4.shape}")

        d4 = self.upsample4(e4)
        # print(f"d4: {d4.shape}")
        d4 = torch.cat([d4, e3], dim=1)
        # print(f"d4: {d4.shape}")
        d4 = self.upconv4(d4)
        # print(f"d4: {d4.shape}")

        d3 = self.upsample3(d4)
        # print(f"d3: {d3.shape}")
        d3 = torch.cat([d3, e2], dim=1)
        # print(f"d3: {d3.shape}")
        d3 = self.upconv3(d3)
        # print(f"d3: {d3.shape}")

        d2 = self.upsample2(d3)
        # print(f"d2: {d2.shape}")
        d2 = torch.cat([d2, e1], dim=1)
        # print(f"d2: {d2.shape}")
        d2 = self.upconv2(d2)
        # print(f"d2: {d2.shape}")

        d1 = self.upsample1(d2)
        # print(f"d1: {d1.shape}")
        d1 = self.upconv1(d1)
        # print(f"d1: {d1.shape}")
        out = self.out(d1)
        # print(f"out: {out.shape}")
        return out

class U_NeXt_v3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4,),
            LayerNorm(64, eps=1e-6, data_format="channels_first"),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
        )
        self.downsample1 = Downsample(64, 128)
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
        )
        self.downsample2 = Downsample(128, 256)
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
        )
        self.downsample3 = Downsample(256, 512)
        self.stage4 = nn.Sequential(
            ConvNeXtBlock(512),
            ConvNeXtBlock(512),
        )
        self.norm = nn.LayerNorm(512, eps=1e-6)
        self.upsample4 = Up_conv(512, 256)
        self.upsample3 = Up_conv(256, 128)
        self.upsample2 = Up_conv(128, 64)
        self.upsample1 = Up_conv(64, 32, scale_factor=4)
        self.out = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.upconv4 = conv_block(512, 256)
        self.upconv3 = conv_block(256, 128)
        self.upconv2 = conv_block(128, 64)
        self.upconv1 = conv_block(32, 32)

    
    def forward(self, x):
        e1 = self.stem(x)
        # print(f"e1: {e1.shape}")
        e1 = self.stage1(e1)
        # print(f"e1: {e1.shape}")
        e2 = self.downsample1(e1)
        # print(f"e2: {e2.shape}")
        e2 = self.stage2(e2)
        # print(f"e2: {e2.shape}")
        e3 = self.downsample2(e2)
        # print(f"e3: {e3.shape}")
        e3 = self.stage3(e3)
        # print(f"e3: {e3.shape}")
        e4 = self.downsample3(e3)
        # print(f"e4: {e4.shape}")
        e4 = self.stage4(e4)
        # print(f"e4: {e4.shape}")

        d4 = self.upsample4(e4)
        # print(f"d4: {d4.shape}")
        d4 = torch.cat([d4, e3], dim=1)
        # print(f"d4: {d4.shape}")
        d4 = self.upconv4(d4)
        # print(f"d4: {d4.shape}")

        d3 = self.upsample3(d4)
        # print(f"d3: {d3.shape}")
        d3 = torch.cat([d3, e2], dim=1)
        # print(f"d3: {d3.shape}")
        d3 = self.upconv3(d3)
        # print(f"d3: {d3.shape}")

        d2 = self.upsample2(d3)
        # print(f"d2: {d2.shape}")
        d2 = torch.cat([d2, e1], dim=1)
        # print(f"d2: {d2.shape}")
        d2 = self.upconv2(d2)
        # print(f"d2: {d2.shape}")

        d1 = self.upsample1(d2)
        # print(f"d1: {d1.shape}")
        d1 = self.upconv1(d1)
        # print(f"d1: {d1.shape}")
        out = self.out(d1)
        # print(f"out: {out.shape}")
        return out

class U_NeXt_v4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4,),
            LayerNorm(64, eps=1e-6, data_format="channels_first"),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
        )
        self.downsample1 = Downsample(64, 128)
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
        )
        self.downsample2 = Downsample(128, 256)
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
            ConvNeXtBlock(256),
        )
        self.downsample3 = Downsample(256, 512)
        self.stage4 = nn.Sequential(
            ConvNeXtBlock(512),
            ConvNeXtBlock(512),
            ConvNeXtBlock(512),
        )
        self.norm = nn.LayerNorm(512, eps=1e-6)
        self.upsample4 = Up_conv(512, 256)
        self.upsample3 = Up_conv(256, 128)
        self.upsample2 = Up_conv(128, 64)
        self.upsample1 = Up_conv(64, 32, scale_factor=4)
        self.out = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.upconv4 = conv_block(512, 256)
        self.upconv3 = conv_block(256, 128)
        self.upconv2 = conv_block(128, 64)
        self.upconv1 = conv_block(32, 32)

    
    def forward(self, x):
        e1 = self.stem(x)
        # print(f"e1: {e1.shape}")
        e1 = self.stage1(e1)
        # print(f"e1: {e1.shape}")
        e2 = self.downsample1(e1)
        # print(f"e2: {e2.shape}")
        e2 = self.stage2(e2)
        # print(f"e2: {e2.shape}")
        e3 = self.downsample2(e2)
        # print(f"e3: {e3.shape}")
        e3 = self.stage3(e3)
        # print(f"e3: {e3.shape}")
        e4 = self.downsample3(e3)
        # print(f"e4: {e4.shape}")
        e4 = self.stage4(e4)
        # print(f"e4: {e4.shape}")

        d4 = self.upsample4(e4)
        # print(f"d4: {d4.shape}")
        d4 = torch.cat([d4, e3], dim=1)
        # print(f"d4: {d4.shape}")
        d4 = self.upconv4(d4)
        # print(f"d4: {d4.shape}")

        d3 = self.upsample3(d4)
        # print(f"d3: {d3.shape}")
        d3 = torch.cat([d3, e2], dim=1)
        # print(f"d3: {d3.shape}")
        d3 = self.upconv3(d3)
        # print(f"d3: {d3.shape}")

        d2 = self.upsample2(d3)
        # print(f"d2: {d2.shape}")
        d2 = torch.cat([d2, e1], dim=1)
        # print(f"d2: {d2.shape}")
        d2 = self.upconv2(d2)
        # print(f"d2: {d2.shape}")

        d1 = self.upsample1(d2)
        # print(f"d1: {d1.shape}")
        d1 = self.upconv1(d1)
        # print(f"d1: {d1.shape}")
        out = self.out(d1)
        # print(f"out: {out.shape}")
        return out
