## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


##########################################################################
## Layer Norm


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
# 在 Attention 类后添加 WindowAttention 类
class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # 定义相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # 获取窗口内每个token的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        # print(f"WindowAttention: input x shape (B_, N, C): {B_, N, C}")
        # print(f"WindowAttention: self.num_heads: {self.num_heads}")
        # print(f"WindowAttention: self.window_size: {self.window_size}")

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # print(f"WindowAttention: attn shape before mask: {attn.shape}") # Should be (B_, self.num_heads, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # print(f"WindowAttention: mask shape: {mask.shape}")
            # print(f"WindowAttention: nW from mask: {nW}")
            # print(f"WindowAttention: Attempting to reshape attn to: ({B_ // nW}, {nW}, {self.num_heads}, {N}, {N})")
            # print(f"WindowAttention: Total elements in attn: {attn.numel()}")
            # print(f"WindowAttention: Expected elements for reshape: { (B_ // nW) * nW * self.num_heads * N * N}")
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 替换原始的 TransformerBlock 类
class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        ffn_expansion_factor (float): Ratio of ffn hidden dim to embedding dim.
        bias (bool): If True, add a learnable bias to linear layers.
        LayerNorm_type (str): Type of LayerNorm.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        shift_size=0,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        drop_path=0.0,
    ):
        super(SwinTransformerBlock, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.ffn_expansion_factor = ffn_expansion_factor

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=bias,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.attn_mask = None
        self._current_mask_shape_tag = None

    def calculate_mask(self, x_size):
        # 计算 SW-MSA 的 attention mask
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        b, c, h_og, w_og = x.shape  # 获取原始高度和宽度
        shortcut = x

        # LayerNorm
        x = self.norm1(x)  # b, c, h_og, w_og

        # 计算填充量并进行填充
        pad_b = (self.window_size - h_og % self.window_size) % self.window_size
        pad_r = (self.window_size - w_og % self.window_size) % self.window_size

        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))  # (pad_left, pad_right, pad_top, pad_bottom)

        b, c, h_pad, w_pad = x.shape  # 获取填充后的高度和宽度

        # 转换为 BHWC 格式进行窗口划分
        x = x.permute(0, 2, 3, 1)  # B, H_pad, W_pad, C

        # 如果使用移位窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # 计算 attention mask，使用填充后的 H_pad, W_pad
            current_shape_tag_for_mask = (h_pad, w_pad)
            if self._current_mask_shape_tag != current_shape_tag_for_mask or self.attn_mask is None:
                # Debug print: 确认这里触发了 mask 的重新计算
                print(f"SwinTransformerBlock: Recalculating mask for shape: {current_shape_tag_for_mask}")
                self.attn_mask = self.calculate_mask((h_pad, w_pad)).to(x.device)
                self._current_mask_shape_tag = current_shape_tag_for_mask
                # Debug print: 确认计算后的 mask 维度
                print(f"SwinTransformerBlock: New mask shape: {self.attn_mask.shape}")
        else:
            shifted_x = x
            self.attn_mask = None

        # 窗口分割 (使用填充后的 H_pad, W_pad)
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # 合并窗口 (使用填充后的 H_pad, W_pad)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h_pad, w_pad)  # B, H_pad, W_pad, C

        # 如果使用了移位窗口，反向移位恢复原始位置
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 转换回BCHW格式
        x = x.permute(0, 3, 1, 2)  # B, C, H_pad, W_pad

        # 如果进行了填充，裁剪回原始尺寸
        if pad_b > 0 or pad_r > 0:
            x = x[:, :, :h_og, :w_og]  # B, C, H_og, W_og

        # FFN
        # shortcut 的尺寸是 (B, C, H_og, W_og)
        # x 此刻的尺寸也是 (B, C, H_og, W_og)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

    # 原forward
    # def forward(self, x):
    #     b, c, h, w = x.shape
    #     shortcut = x

    #     # LayerNorm
    #     x = self.norm1(x)  # b, c, h, w

    #     # 转换为 BHWC 格式进行窗口划分
    #     x = x.permute(0, 2, 3, 1)  # B, H, W, C

    #     # 如果使用移位窗口
    #     if self.shift_size > 0:
    #         shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    #         # 计算 attention mask
    #         if self.attn_mask is None:
    #             self.attn_mask = self.calculate_mask((h, w)).to(x.device)
    #     else:
    #         shifted_x = x
    #         self.attn_mask = None

    #     # 窗口分割
    #     x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    #     x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nW*B, window_size*window_size, C

    #     # W-MSA/SW-MSA
    #     attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    #     # 合并窗口
    #     attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
    #     shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # B, H, W, C

    #     # 如果使用了移位窗口，反向移位恢复原始位置
    #     if self.shift_size > 0:
    #         x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    #     else:
    #         x = shifted_x

    #     # 转换回BCHW格式
    #     x = x.permute(0, 3, 1, 2)  # B, C, H, W

    #     # FFN
    #     x = shortcut + self.drop_path(x)
    #     x = x + self.drop_path(self.ffn(self.norm2(x)))

    #     return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
        dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        window_size=8,
    ):
        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    window_size=window_size,
                    shift_size=0,
                    # shift_size=0 if (i % 2 == 0) else window_size // 2,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    window_size=window_size,
                    shift_size=0,
                    # shift_size=0 if (i % 2 == 0) else window_size // 2,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    window_size=window_size,
                    shift_size=0,
                    # shift_size=0 if (i % 2 == 0) else window_size // 2,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    window_size=window_size,
                    shift_size=0,
                    # shift_size=0 if (i % 2 == 0) else window_size // 2,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    window_size=window_size,
                    shift_size=0,
                    # shift_size=0 if (i % 2 == 0) else window_size // 2,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    window_size=window_size,
                    shift_size=0,
                    # shift_size=0 if (i % 2 == 0) else window_size // 2,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    window_size=window_size,
                    # shift_size=0,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    window_size=window_size,
                    shift_size=0,
                    # shift_size=0 if (i % 2 == 0) else window_size // 2,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == "__main__":
    from torchprofile import profile_macs
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print(f"Using device: {device}")

    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )

    # Move model to the selected device (GPU if available)
    model.to(device)

    dummy_input = torch.randn(1, 3, 128, 128)

    # Move dummy_input to the selected device (GPU if available)
    dummy_input = dummy_input.to(device)

    # Profile MACs - torchprofile should handle the device correctly if inputs are on device
    macs = profile_macs(model, dummy_input)

    # Calculate parameters - this is independent of device, but good to do after moving the model
    params = sum(p.numel() for p in model.parameters())

    print("Restormer Model:")
    print(f"MACs: {macs / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")
