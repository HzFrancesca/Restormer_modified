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


# Attentions
##########################################################################
# HW x HW
class HWxHW_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head (h w) c", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head (h w) c", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head (h w) c", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,h*w,h*w
        attn = attn.softmax(dim=-1)

        out = attn @ v  # b,heads,h*w,c//heads
        out = rearrange(out, "b head (h w) c -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    @staticmethod
    def HWxHW_macs(module, inputs, output):
        """
        Calculates MACs within HWxHW_Attention and adds them to module.total_ops.
        """

        x = inputs[0]
        b, dim, h, w = x.shape
        num_heads = module.num_heads
        c_head = dim // num_heads
        N_seq = h * w

        # Calculate MACs for Conv2d layers
        # 1. MACs for qkv: input (b, dim, h, w), output (b, dim*3, h, w)
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w
        # 2. MACs for qkv_dwconv: input (b, dim*3, h, w), output (b, dim*3, h, w)
        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w
        # 3. MACs for project_out: input (b, dim, h, w), output (b, dim, h, w)
        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w
        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out

        # 矩阵相乘，中间的维度算一个
        # 1. MACs for attn_scores = (q @ k.transpose(-2, -1))
        # q: (b, num_heads, N_seq, c_head)
        # k.transpose: (b, num_heads, c_head, N_seq)
        macs_attn_qk = b * num_heads * N_seq * c_head * N_seq
        # 2. MACs for temperature multiplication: attn = attn_scores * self.temperature
        # attn_scores: (b, num_heads, N_seq, N_seq)
        # self.temperature: (num_heads, 1, 1) - broadcasted to (b, num_heads, N_seq, N_seq)
        macs_temperature = b * num_heads * N_seq * N_seq
        # 3. MACs for out_attn = attn_probs @ v
        # attn_probs: (b, num_heads, N_seq, N_seq) (shape of operand1)
        # v: (b, num_heads, N_seq, c_head) (shape of operand2)
        macs_attn_v = b * num_heads * N_seq * N_seq * c_head
        macs_attn = macs_attn_qk + macs_temperature + macs_attn_v

        macs_all = macs_conv2d + macs_attn

        module.total_ops += torch.DoubleTensor([int(macs_all)])


# C X C
class CxC_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,c//heads,c//heads
        attn = attn.softmax(dim=-1)

        out = attn @ v  # b,heads,c//heads,h*w
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    @staticmethod
    def CxC_macs(module, inputs, output):
        """
        Calculates MACs within CxC_Attention and adds them to module.total_ops.
        """
        x = inputs[0]
        # print(f"Input shape: {x.shape}")
        b, dim, h, w = x.shape
        num_heads = module.num_heads
        c_head = dim // num_heads
        N_seq = h * w

        # 1.Calculate MACs for Conv2d layers
        # For qkv: input (b, dim, h, w), output (b, dim*3, h, w)
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w
        # print("---- Part1:Conv2d ----")
        # print(f"Macs_qkv:{macs_qkv}")
        # For qkv_dwconv: input (b, dim*3, h, w), output (b, dim*3, h, w)
        # Note: Depthwise conv MACs are typically in_channels * k_h * k_w * H_out * W_out
        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w
        # print(f"Macs_qkv_dwconv:{macs_qkv_dwconv}")
        # For project_out: input (b, dim, h, w), output (b, dim, h, w)
        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w
        # print(f"Macs_project_out:{macs_project_out}")
        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out

        # print(f"Macs_conv2d:{macs_conv2d}")

        # # 2.Calculate Normalization
        # macs_norm_q = b * num_heads * c_head * N_seq * 3
        # macs_norm_k = b * num_heads * c_head * N_seq * 3
        # macs_norm = macs_norm_q + macs_norm_k
        # print("---- Part2:Normalizaton ----")
        # print(f"Macs_norm:{macs_norm}")

        # 3.Calculate Attention
        # q @ k.transpose(-2, -1))
        # q: (b, num_heads, c_head, N_seq)
        # k.transpose: (b, num_heads, N_seq, c_head)
        macs_attn_qk = b * num_heads * c_head * N_seq * c_head
        # attn_scores * self.temperature
        # self.temperature: (num_heads, 1, 1) - broadcasted to (b, num_heads, c_head, c_head)
        macs_temperature = b * num_heads * c_head * c_head
        # attn_probs @ v
        # attn_probs: (b, num_heads, c_head, c_head) (shape of operand1)
        # v: (b, num_heads, c_head, N_seq) (shape of operand2)
        macs_attn_v = b * num_heads * c_head * c_head * N_seq
        macs_attn = macs_attn_qk + macs_temperature + macs_attn_v
        # print("---- Part3:Attention ----")
        # print(f"Macs_attn:{macs_attn}")

        # # 4.Calculate Softmax
        # macs_softmax = b * num_heads * c_head * c_head * 3
        # print("---- Part4:Softmax ----")
        # print(f"Macs_softmax:{macs_softmax}")
        # Sum
        # macs_for_matmul_ops = macs_conv2d + macs_norm + macs_attn + macs_softmax
        macs_for_matmul_ops = macs_conv2d + macs_attn
        # print("----- Total -----")
        # print(f"Macs_for_matual_ops:{macs_for_matmul_ops}")
        # print(f"Module.total_ops_before:{module.total_ops.item()}")
        module.total_ops += torch.DoubleTensor([int(macs_for_matmul_ops)])
        # print(f"Module.total_ops_after:{module.total_ops.item()}\n")


# CH x CH
class CHxCH_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head (c h) w", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head (c h) w", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head (c h) w", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,c*h//heads,c*h//heads
        attn = attn.softmax(dim=-1)

        out = attn @ v  # b,heads,c*h//heads,w

        out = rearrange(out, "b head (c h) w -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    @staticmethod
    def CHxCH_macs(module, inputs, output):
        """
        Calculates MACs  within CHxCH_Attention.Adds them to module.total_ops.
        """
        x = inputs[0]
        b, dim, h, w = x.shape
        num_heads = module.num_heads
        c_head = dim // num_heads
        N_seq_q = c_head * h
        N_seq_k = w

        # Conv2d MACs
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w

        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w

        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w

        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out

        # Matrix multiplication MACs
        macs_attn_qk = b * num_heads * N_seq_q * N_seq_k * N_seq_q
        macs_temperature = b * num_heads * N_seq_q * N_seq_q
        macs_attn_v = b * num_heads * N_seq_q * N_seq_q * N_seq_k

        macs_matmul = macs_attn_qk + macs_temperature + macs_attn_v

        # Total MACs
        macs_all = macs_conv2d + macs_matmul

        module.total_ops += torch.DoubleTensor([int(macs_all)])


# WxW
# HTA(Height-stack transposed attention)
class WxW_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head w (c h)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head w (c h)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head w (c h)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,w,w
        attn = attn.softmax(dim=-1)
        out = attn @ v
        # b,heads,w,c*h//heads
        out = rearrange(out, "b head w (c h) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    @staticmethod
    def WxW_macs(module, inputs, output):
        """
        Calculates MACs  within WxW_Attention and adds them to module.total_ops.
        """

        x = inputs[0]
        b, dim, h, w = x.shape

        num_heads = module.num_heads
        c_head = dim // num_heads
        c_h = c_head * h

        # Conv2d MACs
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w
        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w
        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w
        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out

        # Attention Matrix
        macs_attn_qk = b * num_heads * w * c_h * w
        macs_temperature = b * num_heads * w * w
        macs_attn_v = b * num_heads * w * w * c_h
        macs_attn = macs_attn_qk + macs_temperature + macs_attn_v

        macs_all = macs_conv2d + macs_attn
        module.total_ops += torch.DoubleTensor([int(macs_all)])


# CW x CW
class CWxCW_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head (c w) h", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head (c w) h", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head (c w) h", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,c*w//heads,c*w//heads
        attn = attn.softmax(dim=-1)

        out = attn @ v
        # b,heads,c*w//heads,h
        out = rearrange(out, "b head (c w) h-> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    @staticmethod
    def CWxCW_macs(module, inputs, output):
        """
        Calculates MACs  within CWxCW_Attention and adds them to module.total_ops
        """

        x = inputs[0]
        b, dim, h, w = x.shape
        num_heads = module.num_heads
        c_head = dim // num_heads
        N_seq_cw = c_head * w
        N_seq_h = h

        # Conv2d MACs
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w
        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w
        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w
        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out

        # Attention Matrix
        macs_attn_qk = b * num_heads * N_seq_cw * N_seq_h * N_seq_cw
        macs_temperature = b * num_heads * N_seq_cw * N_seq_cw
        macs_attn_v = b * num_heads * N_seq_cw * N_seq_cw * N_seq_h
        macs_attn = macs_attn_qk + macs_temperature + macs_attn_v

        macs_all = macs_conv2d + macs_attn
        module.total_ops += torch.DoubleTensor([int(macs_all)])


# H x H
# Width-stack transposed-attention(WTA)
class HxH_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head h (c w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,h,h
        attn = attn.softmax(dim=-1)

        out = attn @ v
        # b,heads,h,(c*w)//heads

        out = rearrange(out, "b head h (c w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    @staticmethod
    def HxH_macs(module, inputs, output):
        """
        Calculates MACs  within HxH_Attention and adds them to module.total_ops.
        """
        x = inputs[0]
        b, dim_in, h, w = x.shape
        num_heads = module.num_heads
        c_head = dim_in // num_heads

        # Conv2d MACs
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w
        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w
        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w
        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out

        # Attention
        macs_attn_qk = b * num_heads * h * (c_head * w) * h
        macs_temperature = b * num_heads * h * h
        macs_attn_v = b * num_heads * h * h * (c_head * w)
        macs_attn = macs_attn_qk + macs_temperature + macs_attn_v

        macs_all = macs_conv2d + macs_attn
        module.total_ops += torch.DoubleTensor([int(macs_all)])


# C*N*N x C*N*N
# Inter-channel block cross-attention(ICS)
class CNNxCNN_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, blocks=4):
        super().__init__()
        self.N = blocks
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape
        h_pad = self.N - h % self.N if not h % self.N == 0 else 0
        w_pad = self.N - w % self.N if not w % self.N == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), "reflect")

        _, _, h_after, w_after = x.shape

        # 验证
        h_temp = h_after // self.N
        w_temp = w_after // self.N
        assert h_temp * self.N == h + h_pad, "高度分块计算错误"
        assert w_temp * self.N == w + w_pad, "宽度分块计算错误"

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(
            q, "b (head c) (h1 N_h) (w1 N_w) -> b head (c N_h N_w) (h1 w1)", head=self.num_heads, N_h=self.N, N_w=self.N
        )
        k = rearrange(
            k, "b (head c) (h1 N_h) (w1 N_w) -> b head (c N_h N_w) (h1 w1)", head=self.num_heads, N_h=self.N, N_w=self.N
        )
        v = rearrange(
            v, "b (head c) (h1 N_h) (w1 N_w) -> b head (c N_h N_w) (h1 w1)", head=self.num_heads, N_h=self.N, N_w=self.N
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,c*N*N//heads,c*N*N//heads
        attn = attn.softmax(dim=-1)

        out = attn @ v
        # b,heads,c*N*N//heads,h*w//(N*N)
        out = rearrange(
            out,
            "b head (c N_h N_w) (h1 w1) -> b (head c) (h1 N_h) (w1 N_w)",
            head=self.num_heads,
            N_h=self.N,
            N_w=self.N,
            h1=h_after // self.N,
            w1=w_after // self.N,
        )

        out = self.project_out(out)
        # 移除填充，切片回原始维度 (h, w)
        out = out[:, :, :h, :w]
        return out

    @staticmethod
    def CNNxCNN_macs(module, inputs, output):
        """
        Calculates MACs within CNNxCNN_Attentionand adds them to module.total_ops.
        """

        x = inputs[0]
        b, dim, h, w = x.shape
        num_heads = module.num_heads
        N = module.N

        # Calculate padding
        h_pad = N - h % N if not h % N == 0 else 0
        w_pad = N - w % N if not w % N == 0 else 0
        h_after = h + h_pad
        w_after = w + w_pad

        c_head = dim // num_heads
        c_block = c_head * N * N  # channels per block (c*N*N//num_heads)
        N_seq = (h_after * w_after) // (N * N)  # sequence length (h*w//(N*N))

        # Conv2d MACs
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w
        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w
        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w
        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out

        # Attention
        macs_attn_qk = b * num_heads * c_block * N_seq * c_block
        macs_temperature = b * num_heads * c_block * c_block
        macs_attn_v = b * num_heads * c_block * c_block * N_seq
        macs_attn = macs_attn_qk + macs_temperature + macs_attn_v

        macs_all = macs_conv2d + macs_attn
        module.total_ops += torch.DoubleTensor([int(macs_all)])


# Intra-channel blocks:C x N*N x N*N
class CxNNxNN_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, blocks=8):
        super().__init__()
        self.N = blocks
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape
        h_pad = self.N - h % self.N if not h % self.N == 0 else 0
        w_pad = self.N - w % self.N if not w % self.N == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), "reflect")

        _, _, h_after, w_after = x.shape

        # 验证
        h_temp = h_after // self.N
        w_temp = w_after // self.N
        assert h_temp * self.N == h + h_pad, "高度分块计算错误"
        assert w_temp * self.N == w + w_pad, "宽度分块计算错误"

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(
            q, "b (head c) (h1 N_h) (w1 N_w) -> b head c (N_h N_w) (h1 w1)", head=self.num_heads, N_h=self.N, N_w=self.N
        )
        k = rearrange(
            k, "b (head c) (h1 N_h) (w1 N_w) -> b head c (N_h N_w) (h1 w1)", head=self.num_heads, N_h=self.N, N_w=self.N
        )
        v = rearrange(
            v, "b (head c) (h1 N_h) (w1 N_w) -> b head c (N_h N_w) (h1 w1)", head=self.num_heads, N_h=self.N, N_w=self.N
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,c//heads,N*N,N*N
        attn = attn.softmax(dim=-1)

        out = attn @ v
        # b,heads,c//heads,N*N,h*w//(N*N)
        out = rearrange(
            out,
            "b head c (N_h N_w) (h1 w1) -> b (head c) (h1 N_h) (w1 N_w)",
            head=self.num_heads,
            N_h=self.N,
            N_w=self.N,
            h1=h_after // self.N,
            w1=w_after // self.N,
        )

        out = self.project_out(out)
        # 移除填充，切片回原始维度 (h, w)
        out = out[:, :, :h, :w]
        return out

    @staticmethod
    def CxNNxNN_macs(module, inputs, output):
        """Calculates MACs  within CxNNxNN_Attention and adds them to module.total_ops"""
        x = inputs[0]
        b, dim_in, h, w = x.shape
        N = module.N
        num_heads = module.num_heads

        # 计算填充后大小
        h_pad = N - h % N if h % N != 0 else 0
        w_pad = N - w % N if w % N != 0 else 0
        h_after = h + h_pad
        w_after = w + w_pad

        c_head = dim_in // num_heads
        h1 = h_after // N
        w1 = w_after // N

        # Conv2d MACs
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w
        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w
        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w
        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out

        # Attention
        macs_attn_qk = b * num_heads * c_head * (N * N) * (h1 * w1) * (N * N)
        macs_temperature = b * num_heads * c_head * (N * N) * (N * N)
        macs_attn_v = b * num_heads * c_head * (N * N) * (N * N) * (h1 * w1)
        macs_attn = macs_attn_qk + macs_temperature + macs_attn_v

        macs_all = macs_conv2d + macs_attn
        module.total_ops += torch.DoubleTensor([int(macs_all)])


# C x H x H
# Intra-channel row attention(IRS)
class CxHxH_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c h w", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c h w", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c h w", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,c//heads,h,h
        attn = attn.softmax(dim=-1)

        out = attn @ v  # b,heads,c//heads,h,w
        out = rearrange(out, "b head c h w -> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    @staticmethod
    def CxHxH_macs(module, inputs, output):
        """
        Calculates MACs within CxHxH_Attention and adds them to module.total_ops.
        """

        x = inputs[0]
        b, dim, h, w = x.shape
        num_heads = module.num_heads
        c_head = dim // num_heads

        # Conv2d MACs
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w
        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w
        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w
        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out
        # Attention
        macs_attn_qk = b * num_heads * c_head * h * w * h
        macs_temperature = b * num_heads * c_head * h * h
        macs_attn_v = b * num_heads * c_head * h * h * w
        macs_attn = macs_attn_qk + macs_temperature + macs_attn_v

        macs_all = macs_conv2d + macs_attn
        module.total_ops += torch.DoubleTensor([int(macs_all)])


# C x W x W
# Intra-channel column(ICS)
class CxWxW_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c w h", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c w h", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c w h", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # b,heads,c//heads,w,w
        attn = attn.softmax(dim=-1)

        out = attn @ v  # b,heads,c//heads,w,h
        out = rearrange(out, "b head c w h-> b (head c) h w", head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    @staticmethod
    def CxWxW_macs(module, inputs, output):
        x = inputs[0]
        b, dim_in, h, w = x.shape
        num_heads = module.num_heads
        c_head = dim_in // num_heads

        # Conv2d MACs
        macs_qkv = (
            module.qkv.in_channels
            * module.qkv.out_channels
            * module.qkv.kernel_size[0]
            * module.qkv.kernel_size[1]
            * h
            * w
        )
        if module.qkv.bias is not None:
            macs_qkv += module.qkv.out_channels * h * w
        macs_qkv_dwconv = (
            (module.qkv_dwconv.in_channels // module.qkv_dwconv.groups)
            * module.qkv_dwconv.out_channels
            * module.qkv_dwconv.kernel_size[0]
            * module.qkv_dwconv.kernel_size[1]
            * h
            * w
        )
        if module.qkv_dwconv.bias is not None:
            macs_qkv_dwconv += module.qkv_dwconv.out_channels * h * w
        macs_project_out = (
            module.project_out.in_channels
            * module.project_out.out_channels
            * module.project_out.kernel_size[0]
            * module.project_out.kernel_size[1]
            * h
            * w
        )
        if module.project_out.bias is not None:
            macs_project_out += module.project_out.out_channels * h * w
        macs_conv2d = macs_qkv + macs_qkv_dwconv + macs_project_out

        # Attention
        macs_attn_qk = b * num_heads * c_head * w * h * w
        macs_temperature = b * num_heads * c_head * w * w
        macs_attn_v = b * num_heads * c_head * w * w * h

        macs_attn = macs_attn_qk + macs_temperature + macs_attn_v

        macs_all = macs_conv2d + macs_attn
        module.total_ops += torch.DoubleTensor([int(macs_all)])


# TransformerBlocks
##########################################################################
class HWxHW_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = HWxHW_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class CxC_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CxC_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class CHxCH_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CHxCH_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class WxW_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = WxW_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class CWxCW_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CWxCW_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class HxH_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = HxH_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class CNNxCNN_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CNNxCNN_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class CxNNxNN_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CxNNxNN_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class CxHxH_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CxHxH_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class CxWxW_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CxWxW_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class HTA_ICST(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = WxW_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn2 = CxWxW_Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x))
        x = x + self.ffn(self.norm3(x))

        return x


class WTA_IRST(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = HxH_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn2 = CxHxH_Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x))
        x = x + self.ffn(self.norm3(x))

        return x


class DST(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.layer1 = HTA_ICST(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.layer2 = WTA_IRST(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class IBCT(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.layer1 = CNNxCNN_TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.layer2 = CNNxCNN_TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
