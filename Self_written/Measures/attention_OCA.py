import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from attentions import LayerNorm, FeedForward


def to(x):
    return {"device": x.device, "dtype": x.dtype}


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum("b x y d, r d -> b x y r", q, rel_k)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head**-0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, "b (x y) c -> b x y c", x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b x i y j-> b (x y) (i j)")

        q = rearrange(q, "b x y d -> b y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b x i y j -> b (y x) (j i)")
        return rel_logits_w + rel_logits_h


##########################################################################
## Overlapping Cross-Attention (OCA)
## 图像size应该被window_size整除，默认值为8
class OCA(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size, overlap_ratio, dim_head):
        super(OCA, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )
        self.qkv = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size=window_size, rel_size=window_size + (self.overlap_win_size - window_size), dim_head=self.dim_head
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        # print(qs.shape, self.window_size)
        qs = rearrange(qs, "b c (h p1) (w p2) -> (b h w) (p1 p2) c", p1=self.window_size, p2=self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, "b (c j) i -> (b i) j c", c=self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        # split heads
        qs, ks, vs = map(
            lambda t: rearrange(t, "b n (head c) -> (b head) n c", head=self.num_spatial_heads), (qs, ks, vs)
        )

        # attention
        qs = qs * self.scale
        spatial_attn = qs @ ks.transpose(-2, -1)
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = spatial_attn @ vs

        out = rearrange(
            out,
            "(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)",
            head=self.num_spatial_heads,
            h=h // self.window_size,
            w=w // self.window_size,
            p1=self.window_size,
            p2=self.window_size,
        )

        # merge spatial and channel(depth-wise conv)
        out = self.project_out(out)

        return out

    def OCA_macs(module, inputs, output):
        """
        Calculates MACs within OCA and adds them to module.total_ops.
        """
        x = inputs[0]
        b, dim, h, w = x.shape
        num_heads = module.num_spatial_heads
        dim_head = module.dim_head
        inner_dim = module.inner_dim
        window_size = module.window_size
        overlap_win_size = module.overlap_win_size

        # Calculate MACs for Conv2d layers
        # 1. MACs for qkv: input (b, dim, h, w), output (b, inner_dim * 3, h, w)
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

        # 2. MACs for project_out: input (b, inner_dim, h, w), output (b, dim, h, w)
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

        macs_conv2d = macs_qkv + macs_project_out

        # Calculate MACs for attention
        N_seq = window_size * window_size
        overlap_seq = overlap_win_size * overlap_win_size
        num_windows_h = h // window_size
        num_windows_w = w // window_size
        num_windows = num_windows_h * num_windows_w  # Total number of windows

        # Calculate MACs for attention
        N_seq = window_size * window_size
        overlap_seq = overlap_win_size * overlap_win_size

        # 1. MACs for attention scores: qs @ ks.transpose(-2, -1)
        macs_attn_qk = (b * num_windows * num_heads) * N_seq * dim_head * overlap_seq

        # 2. MACs for relative positional embedding addition (Revised)
        rel_size = window_size + (overlap_win_size - window_size)
        rel_dim = rel_size * 2 - 1

        macs_rel_logits_w = (b * num_windows * num_heads) * N_seq * rel_dim * dim_head
        macs_rel_logits_h = (b * num_windows * num_heads) * N_seq * rel_dim * dim_head

        macs_rel_pos_emb_internal = macs_rel_logits_w + macs_rel_logits_h

        # macs_rel_pos_emb_additions = b * num_windows * num_heads * N_seq * overlap_seq

        # 3. MACs for attention output: spatial_attn @ vs
        macs_attn_v = (b * num_windows * num_heads) * N_seq * overlap_seq * dim_head
        macs_attn = macs_attn_qk + macs_rel_pos_emb_internal + macs_attn_v

        # Total MACs
        macs_all = macs_conv2d + macs_attn

        module.total_ops += torch.DoubleTensor([int(macs_all)])


class OCA_TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        window_size=8,
        overlap_ratio=0.5,
        spatial_dim_head=16,
    ):
        super(OCA_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.spatial_attn = OCA(dim, num_heads, bias, window_size, overlap_ratio, spatial_dim_head)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.spatial_attn(self.norm1(x))
        x = x + self.spatial_ffn(self.norm2(x))
        return x
