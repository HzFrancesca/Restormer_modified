## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


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


class FeedForward(nn.Module):  # SFFN
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * 2.667)
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
        # self.mdcffn = MDCffn(hidden_features * 2, bias=bias)
        # self.mdcffn1 = MDCffn(hidden_features, bias=bias)
        # self.mdcffn2 = MDCffn(hidden_features, bias=bias)
        # self.mdcffn2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
        #                                   groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x1, x2 = self.mdcffn(x).chunk(2, dim=1)
        # x1, x2 = self.project_in(x).chunk(2, dim=1)
        # x1, x2 = self.mdcffn2(x1), self.mdcffn1(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MDC(nn.Module):
    def __init__(self, dim, bias):
        super(MDC, self).__init__()

        self.mdc1 = nn.Conv2d(dim, int(dim / 4), groups=int(dim / 4), kernel_size=3, bias=bias, padding=1)
        self.mdc2 = nn.Conv2d(dim, int(dim / 4), groups=int(dim / 4), kernel_size=5, bias=bias, padding=2)
        self.mdc3 = nn.Conv2d(dim, int(dim / 4), groups=int(dim / 4), kernel_size=7, bias=bias, padding=3)
        self.mdc4 = nn.Conv2d(dim, int(dim / 4), groups=int(dim / 4), kernel_size=9, bias=bias, padding=4)
        self.sig = nn.Sigmoid()
        # self.dconv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x):
        x1 = self.mdc1(x)
        x2 = self.mdc2(x)
        x3 = self.mdc3(x)
        x4 = self.mdc4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.sig(x) * out
        # out = self.dconv3(x)
        return out


class MDCffn(nn.Module):
    def __init__(self, dim, bias):
        super(MDCffn, self).__init__()

        # self.mdc1 = nn.Conv2d(dim, int(dim / 4), groups=int(dim / 4), kernel_size=3, bias=bias, padding=1)
        # self.mdc2 = nn.Conv2d(dim, int(dim / 4), groups=int(dim / 4), kernel_size=5, bias=bias, padding=2)
        # self.mdc3 = nn.Conv2d(dim, int(dim / 4), groups=int(dim / 4), kernel_size=7, bias=bias, padding=3)
        # self.mdc4 = nn.Conv2d(dim, int(dim / 4), groups=int(dim / 4), kernel_size=9, bias=bias, padding=4)
        # self.sig = nn.Sigmoid()
        self.dconv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x):
        # x1 = self.mdc1(x)
        # x2 = self.mdc2(x)
        # x3 = self.mdc3(x)
        # x4 = self.mdc4(x)
        # out = torch.cat([x1, x2, x3, x4], dim=1)
        # out = self.sig(x) * out
        out = self.dconv3(x)
        return out


class HTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(HTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.mdc_q = MDC(dim, bias)
        # self.mdc_k = MDC(dim, bias)
        self.mdc_q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_v = MDC(dim, bias)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = self.mdc_q(q), self.mdc_k(k), self.mdc_v(v)

        q = rearrange(q, "b (head c) h w -> b head (c h) w", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head (c h) w", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head (c h) w", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-2)

        attn = (q.transpose(-2, -1) @ k) * self.temperature  # wxw

        #  图片的注意力
        attn = attn.softmax(dim=-2)

        out = v @ attn

        out = rearrange(out, "b head (c h) w -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class IBC(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(IBC, self).__init__()
        # self.mdc_q = MDC(dim, bias)
        # self.mdc_k = MDC(dim, bias)
        self.mdc_q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_v = MDC(dim, bias)
        self.N = 4
        self.query1 = nn.Conv2d(dim, dim, 1)
        self.key1 = nn.Conv2d(dim, dim, 1)
        self.value1 = nn.Conv2d(dim, dim, 1)
        self.num_heads = num_heads
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, xu, xd):
        # deep_feat: 深层特征 (B, C, H1, W1)  # after up sample
        # shallow_feat: 浅层特征 (B, C, H2, W2)
        n11, c11, h11, w11 = xu.shape
        h_pad = 4 - h11 % 4 if not h11 % 4 == 0 else 0
        w_pad = 4 - w11 % 4 if not w11 % 4 == 0 else 0
        xu_1 = F.pad(xu, (0, w_pad, 0, h_pad), "reflect")
        xd_1 = F.pad(xd, (0, w_pad, 0, h_pad), "reflect")

        _, _, h, w = xu_1.shape
        q1 = self.query1(xu_1)
        k1 = self.key1(xd_1)
        v1 = self.value1(xd_1)
        q1 = self.mdc_q(q1)
        k1 = self.mdc_k(k1)
        v1 = self.mdc_v(v1)
        v = rearrange(
            v1, "b (head c) (h1 h) (w1 w) -> b head (c h1 w1) (h w)", h1=self.N, w1=self.N, head=self.num_heads
        )
        q = rearrange(
            q1, "b (head c) (h1 h) (w1 w) -> b head (c h1 w1) (h w)", h1=self.N, w1=self.N, head=self.num_heads
        )  # N^2,HW/N^2
        k = rearrange(
            k1, "b (head c) (h1 h) (w1 w) -> b head (c h1 w1) (h w)", h1=self.N, w1=self.N, head=self.num_heads
        )
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # CN^2 x CN^2
        attn = attn.softmax(dim=-1)  # 图片内部注意力
        out = attn @ v
        out = rearrange(
            out,
            "b head (c h1 w1) (h w) -> b (head c) (h1 h) (w1 w)",
            h=int(h / self.N),
            w=int(w / self.N),
            h1=self.N,
            w1=self.N,
            head=self.num_heads,
        )
        out = self.project_out(out)
        out = out[:, :, :h11, :w11]
        return out


class WTA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(WTA, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.mdc_q = MDC(dim, bias)
        # self.mdc_k = MDC(dim, bias)
        self.mdc_q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_v = MDC(dim, bias)
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = self.mdc_q(q), self.mdc_k(k), self.mdc_v(v)
        # 计算注意力权重
        v1 = rearrange(v, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        q1 = rearrange(q, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        k1 = rearrange(k, "b (head c) h w -> b head h (c w)", head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        out = attn1 @ v1
        out = rearrange(out, "b head h (c w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        # 特征融合
        out = self.project_out(out)
        return out


class IRS(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(IRS, self).__init__()
        self.temperature = nn.Parameter(torch.ones(dim, 1, 1))
        # self.mdc_q = MDC(dim, bias)
        # self.mdc_k = MDC(dim, bias)
        self.mdc_q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_v = MDC(dim, bias)
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = self.mdc_q(q), self.mdc_k(k), self.mdc_v(v)

        q1 = torch.nn.functional.normalize(q, dim=-1)
        k1 = torch.nn.functional.normalize(k, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        out = attn1 @ v
        # 特征融合
        out = self.project_out(out)
        return out


class ICS(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(ICS, self).__init__()
        self.temperature = nn.Parameter(torch.ones(dim, 1, 1))
        # self.mdc_q = MDC(dim, bias)
        # self.mdc_k = MDC(dim, bias)
        self.mdc_q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.mdc_v = MDC(dim, bias)
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q, k, v = self.mdc_q(q), self.mdc_k(k), self.mdc_v(v)  # CHW

        q1 = torch.nn.functional.normalize(q, dim=-2)
        k1 = torch.nn.functional.normalize(k, dim=-2)
        attn1 = (q1.transpose(-2, -1) @ k1 * self.temperature).softmax(dim=-2)  # CWW
        out = v @ attn1  # CHW@CWW -> CHW
        # 特征融合
        out = self.project_out(out)
        return out


##################################################################################################################


class IBCT(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(IBCT, self).__init__()
        self.attn = IBC(dim, num_heads, bias)
        # self.ffn = SFFN(dim, bias)
        self.ffn = FeedForward(dim, bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)

    def forward(self, xu, xd):  # 其实进来的都是x
        # xun, xdn = self.norm1(xu), self.norm2(xd)
        xun = self.norm1(xu)
        # x = xd + self.attn(xun, xdn)
        x = xd + self.attn(xun, xun)
        x = x + self.ffn(self.norm3(x))

        return x


##########################################################################
class WTAT(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(WTAT, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = SFFN(dim, bias)
        self.ffn = FeedForward(dim, bias)
        self.attn = WTA(dim, num_heads, bias)
        self.attn2 = IRS(dim, num_heads, bias)
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.attn = Attention_inter(dim, num_heads, bias)

    def forward(self, x):
        x_n = self.norm1(x)
        fea1 = x + self.attn(x_n)
        fea2 = fea1 + self.attn2(self.norm3(fea1))
        out = fea2 + self.ffn(self.norm2(fea2))
        return out


############################################################################################
class HTAT(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(HTAT, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = HTA(dim, num_heads, bias)
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.attn = Attention_inter(dim, num_heads, bias)
        self.attn2 = ICS(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = SFFN(dim, bias)
        self.ffn = FeedForward(dim, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.attn2(self.norm3(x))
        x = x + self.ffn(self.norm2(x))

        return x


class DST(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self, n_layers=6, dim=3, num_heads=4, ffn_expansion_factor=2.667, bias=True, LayerNorm_type="WithBias"
    ):
        # 2048
        super().__init__()

        self.layer_stack = nn.ModuleList(
            [
                module
                for _ in range(n_layers)
                for module in (
                    HTAT(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type),  # 先HTAT
                    WTAT(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type),  # 后WTAT
                )
            ]
        )

    def forward(self, x):
        for enc_layer in self.layer_stack:
            x = enc_layer(x)
        return x


class IBCTB(nn.Module):
    """AN encoder model with self attention mechanism."""

    def __init__(
        self, n_layers=4, dim=3, num_heads=4, ffn_expansion_factor=2.667, bias=True, LayerNorm_type="WithBias"
    ):
        super().__init__()
        self.layer_stack = nn.ModuleList(
            [IBCT(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type) for _ in range(2 * n_layers)]
        )

    def forward(self, x):
        for enc_layer in self.layer_stack:
            x = enc_layer(x, x)
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
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2)
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
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=2,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
        dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):
        super(Restormer, self).__init__()
        self.dim = dim
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = IBCTB(
            dim=dim,
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            n_layers=num_blocks[0],
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = DST(
            dim=int(dim * 2**1),
            num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            n_layers=num_blocks[1],
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = DST(
            dim=int(dim * 2**2),
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            n_layers=num_blocks[2],
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        # self.latent = TransformerBlock_1(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[3])

        self.latent = DST(
            dim=int(dim * 2**3),
            num_heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            n_layers=num_blocks[3],
        )

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = DST(
            dim=int(dim * 2**2),
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            n_layers=num_blocks[2],
        )
        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        # self.decoder_level2 = TransformerBlock_1(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[1])
        self.decoder_level2 = DST(
            dim=int(dim * 2**1),
            num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            n_layers=num_blocks[1],
        )

        self.up2_1 = Upsample(int(dim * 2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        # self.decoder_level1 = TransformerBlock_1(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[0])
        self.decoder_level1 = IBCTB(
            dim=int(dim * 2),
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            n_layers=int(num_blocks[0]),
        )

        self.refinement = IBCTB(
            dim=int(dim * 2),
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            n_layers=num_refinement_blocks,
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
        num_blocks=[1, 2, 3, 4],
        num_refinement_blocks=1,
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
