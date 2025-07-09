## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import math
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


##########################################################################


##################################################################
# Axial_attention
class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56, stride=1, bias=False, width=False):
        # in_planes: 输入特征图的通道数 (C_in)
        # out_planes: 输出特征图的通道数 (C_out)
        # groups: 注意力头的数量 (多头注意力机制)
        # kernel_size: 执行注意力的轴的长度 (H 或 W)
        # stride: 如果需要，在输出时应用的步长，用于下采样
        # width: 一个布尔值。如果为 True，则在宽度轴上应用注意力；否则在高度轴上应用。
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        # self.group_planes: 每个注意力头所处理的通道数
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        # 1. 多头自注意力的QKV变换层
        # 使用一个 1x1 卷积将输入特征从 in_planes 映射到 out_planes * 2。
        # 乘以2是因为后续的 Value 部分由内容(content)和位置(position)两部分组成，每一部分维度都是 out_planes。
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1, padding=0, bias=False)
        # 对变换后的 qkv 进行批归一化
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        # 对计算出的相似度矩阵进行批归一化
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        # self.bn_qk = nn.BatchNorm2d(groups)
        # self.bn_qr = nn.BatchNorm2d(groups)
        # self.bn_kr = nn.BatchNorm2d(groups)
        # 对最终的输出进行批归一化
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        # 2. 相对位置编码 (Relative Position Embedding)
        # 这是一个可学习的参数，用于存储相对位置信息。
        # Shape: (group_planes * 2, kernel_size * 2 - 1)
        # 作用:
        #   - 第一维 `group_planes * 2`: 编码了Q, K, V的位置信息。
        #   - 第二维 `kernel_size * 2 - 1`: 覆盖了从 -(kernel_size-1) 到 +(kernel_size-1) 的所有可能的相对距离。
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)

        # 生成用于查询相对位置编码的索引
        query_index = torch.arange(kernel_size).unsqueeze(0)  # Shape: (1, kernel_size)
        key_index = torch.arange(kernel_size).unsqueeze(1)  # Shape: (kernel_size, 1)

        # relative_index[i, j] = j - i，计算query位置j和key位置i之间的相对距离
        # 加上 kernel_size - 1 是为了将索引偏移到非负数范围 [0, 2*kernel_size - 2]
        relative_index = key_index - query_index + kernel_size - 1

        # 将索引矩阵展平，方便后续使用 torch.index_select 高效查找
        # register_buffer: 将一个张量注册为模型的缓冲区，它会随模型保存，但不是模型参数（不会被优化器更新）
        self.register_buffer("flatten_index", relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # Shape: (N, C, H, W)

        # --- 步骤 1: Permute & Reshape, 为1D注意力做准备 ---
        if self.width:
            # 在宽度(W)轴上做注意力，将H视为批次维度的一部分
            x = x.permute(0, 2, 1, 3)  # Shape: (N, H, C, W)
        else:
            # 在高度(H)轴上做注意力，将W视为批次维度的一部分 (以这个分支为例解释)
            x = x.permute(0, 3, 1, 2)  # Shape: (N, W, C, H)
        N, W, C, H = x.shape

        # 将批次维度和空间维度合并，使每一列(或行)成为一个独立的序列
        x = x.contiguous().view(N * W, C, H)
        # Shape: (N*W, C, H)
        # 作用: 将问题转化为一个批次大小为 N*W 的1D序列注意力问题，序列长度为H，特征维度为C。

        # Transformations
        # --- 步骤 2: QKV 变换与分组 ---
        # 通过1x1卷积(qkv_transform)将输入映射到Q,K,V空间，然后进行BN
        qkv = self.bn_qkv(self.qkv_transform(x))
        # Shape of qkv: (N*W, out_planes * 2, H)
        # 作用: 生成了Query, Key, Value的原始数据。

        # 将qkv变形并切分为Q, K, V
        # Reshape: (N*W, groups, group_planes * 2, H)
        # 作用: 将通道维度拆分为 "头" 和 "每个头的通道"
        q, k, v = torch.split(
            qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
            [self.group_planes // 2, self.group_planes // 2, self.group_planes],
            dim=2,
        )
        # Shape of q: (N*W, groups, group_planes/2, H) -> 查询向量
        # Shape of k: (N*W, groups, group_planes/2, H) -> 键向量
        # Shape of v: (N*W, groups, group_planes,   H) -> 值向量

        # Calculate position embedding
        # --- 步骤 3: 计算包含相对位置编码的相似度矩阵 ---
        # 从可学习的 `self.relative` 参数中，根据预先计算好的 `flatten_index` 提取出所有位置对的相对编码
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(
            self.group_planes * 2, self.kernel_size, self.kernel_size
        )
        # Shape of all_embeddings: (group_planes*2, H, H)
        # 作用: 这是一个查找表，`all_embeddings[:, i, j]` 存储了位置i和j之间的相对位置编码。

        # 将位置编码也切分为q, k, v三部分
        q_embedding, k_embedding, v_embedding = torch.split(
            all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0
        )
        # Shape of q_embedding: (group_planes/2, H, H)
        # Shape of k_embedding: (group_planes/2, H, H)
        # Shape of v_embedding: (group_planes,   H, H)

        # 计算注意力分数 (logits)，这里包含三项
        # 1. 内容-内容 (Content-Content): q 和 k 的点积
        qk = torch.einsum("bgci, bgcj->bgij", q, k)
        # Shape of qk: (N*W, groups, H, H)
        # 作用: 标准的自注意力，衡量内容之间的相似性。

        # 2. 内容-位置 (Content-Position): q 和 q_embedding 的点积
        qr = torch.einsum("bgci,cij->bgij", q, q_embedding)
        # Shape of qr: (N*W, groups, H, H)
        # 作用: 衡量每个查询向量q与所有相对位置编码的相似性。

        # 3. 位置-内容 (Position-Content): k 和 k_embedding 的点积
        kr = torch.einsum("bgci,cij->bgij", k, k_embedding).transpose(2, 3)
        # Shape of kr: (N*W, groups, H, H)
        # 作用: 衡量每个键向量k与所有相对位置编码的相似性。转置是为了正确对齐。

        # 将三项相似度拼接、BN、然后求和
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        # Shape: (N*W, groups * 3, H, H)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # .sum(dim=1) 实质上是将三个分离出来的相似度矩阵 (qk, qr, kr 经过BN后的版本) 沿着刚才新创建的、长度为 3 的维度（dim=1）进行逐元素相加。
        # 通过这个求和操作，模型将内容信息 (qk) 和两种位置信息 (qr, kr) 融合在了一起，形成了一个统一的、最终的注意力分数矩阵。
        # Shape: (N*W, groups, H, H)
        # 作用: 合并了内容和位置信息，得到最终的注意力分数

        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)

        # --- 步骤 4: Softmax 和加权求和 ---
        # 沿key的维度(dim=3, 即最后一个H)进行softmax，得到注意力权重，即ttention Matrix
        similarity = F.softmax(stacked_similarity, dim=3)
        # Shape: (N*W, groups, H, H)
        # 作用: `similarity[b, g, i, j]` 表示第i个输出对第j个输入的注意力权重。

        # 使用注意力权重对 value 进行加权求和
        # 1. 内容输出
        sv = torch.einsum("bgij,bgcj->bgci", similarity, v)
        # Shape of sv: (N*W, groups, group_planes, H)
        # 作用: 这是标准的注意力输出，是内容信息的加权聚合。

        # 2. 位置输出
        sve = torch.einsum("bgij,cij->bgci", similarity, v_embedding)
        # Shape of sve: (N*W, groups, group_planes, H)
        # 作用: 这是位置编码的加权聚合，为输出增加了位置偏差。

        # --- 步骤 5: 输出合并与最终塑形 ---
        # 将内容输出和位置输出拼接起来
        # 注意：这里的 dim=-1 是沿着最后一个维度(H)拼接，这是一个比较特殊的操作。
        # 更常见的做法是沿着通道维度(dim=2)拼接。但从后续的view来看，这里的维度计算是匹配的。
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        # Shape: (N*W, out_planes * 2, H)

        # 经过BN，然后通过 view + sum 的方式将两个流(sv, sve)的信息合并
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)
        # Shape after view: (N, W, out_planes, 2, H)
        # Shape after sum: (N, W, out_planes, H)
        # 作用: 得到每个注意力头的最终输出，并将通道数恢复到 out_planes。

        # --- 步骤 6: Permute Back & Pooling ---
        # 将Tensor的形状恢复到 (N, C, H, W) 的格式
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        # 如果stride > 1, 进行平均池化下采样
        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1.0 / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0.0, math.sqrt(1.0 / self.group_planes))


class Axial_TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        stride=1,
        groups=1,
        kernel_size=56,
    ):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.hight_block = AxialAttention(dim, dim, groups=groups, kernel_size=kernel_size, stride=stride, width=False)
        self.width_block = AxialAttention(dim, dim, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.width_block(self.hight_block(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))

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
    ):
        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                Axial_TransformerBlock(
                    dim=dim,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stride=1,
                    groups=heads[0],
                    kernel_size=56,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[
                Axial_TransformerBlock(
                    dim=int(dim * 2**1),
                    groups=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stride=1,
                    kernel_size=28,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[
                Axial_TransformerBlock(
                    dim=int(dim * 2**2),
                    groups=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stride=1,
                    kernel_size=14,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[
                Axial_TransformerBlock(
                    dim=int(dim * 2**3),
                    groups=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stride=1,
                    kernel_size=7,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[
                Axial_TransformerBlock(
                    dim=int(dim * 2**2),
                    groups=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stride=1,
                    kernel_size=14,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[
                Axial_TransformerBlock(
                    dim=int(dim * 2**1),
                    groups=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stride=1,
                    kernel_size=28,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[
                Axial_TransformerBlock(
                    dim=int(dim * 2**1),
                    groups=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stride=1,
                    kernel_size=56,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                Axial_TransformerBlock(
                    dim=int(dim * 2**1),
                    groups=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stride=1,
                    kernel_size=56,
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
