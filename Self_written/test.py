import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat  # Make sure einops is installed (`pip install einops`)


# Include the class definition here
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
        print(f"--- Inside forward pass ---")
        print(f"Input x shape: {x.shape}")  # (B, C_in, H, W)

        _, _, h, w = x.shape
        h_pad = self.N - h % self.N if not h % self.N == 0 else 0
        w_pad = self.N - w % self.N if not w % self.N == 0 else 0

        x_padded = F.pad(x, (0, w_pad, 0, h_pad), "reflect")
        print(f"x shape after padding: {x_padded.shape}")  # (B, C_in, H_after, W_after)

        _, _, h_after, w_after = x_padded.shape

        h_temp = h_after // self.N
        w_temp = w_after // self.N
        assert h_temp * self.N == h + h_pad, "高度分块计算错误"
        assert w_temp * self.N == w + w_pad, "宽度分块计算错误"
        print(f"Padded dimensions: H_after={h_after}, W_after={w_after}. Blocks H_1={h_temp}, W_1={w_temp}")

        qkv_proj = self.qkv(x_padded)
        print(f"qkv shape after 1x1 conv: {qkv_proj.shape}")  # (B, C_in*3, H_after, W_after)

        qkv = self.qkv_dwconv(qkv_proj)
        print(f"qkv shape after 3x3 dwconv: {qkv.shape}")  # (B, C_in*3, H_after, W_after)

        q, k, v = qkv.chunk(3, dim=1)
        print(f"q shape after chunk: {q.shape}")  # (B, C_in, H_after, W_after)
        print(f"k shape after chunk: {k.shape}")  # (B, C_in, H_after, W_after)
        print(f"v shape after chunk: {v.shape}")  # (B, C_in, H_after, W_after)

        q_rearranged = rearrange(
            q, "b (head c) (h1 N_h) (w1 N_w) -> b head c (N_h N_w) (h1 w1)", head=self.num_heads, N_h=self.N, N_w=self.N
        )
        print(f"q shape after rearrange: {q_rearranged.shape}")  # (B, N_h, C_h, N*N, H_1*W_1)

        k_rearranged = rearrange(
            k, "b (head c) (h1 N_h) (w1 N_w) -> b head c (N_h N_w) (h1 w1)", head=self.num_heads, N_h=self.N, N_w=self.N
        )
        print(f"k shape after rearrange: {k_rearranged.shape}")  # (B, N_h, C_h, N*N, H_1*W_1)

        v_rearranged = rearrange(
            v, "b (head c) (h1 N_h) (w1 N_w) -> b head c (N_h N_w) (h1 w1)", head=self.num_heads, N_h=self.N, N_w=self.N
        )
        print(f"v shape after rearrange: {v_rearranged.shape}")  # (B, N_h, C_h, N*N, H_1*W_1)

        q_norm = F.normalize(q_rearranged, dim=-1)
        print(f"q shape after normalize: {q_norm.shape}")  # (B, N_h, C_h, N*N, H_1*W_1)

        k_norm = F.normalize(k_rearranged, dim=-1)
        print(f"k shape after normalize: {k_norm.shape}")  # (B, N_h, C_h, N*N, H_1*W_1)

        # Transpose k for matrix multiplication
        k_transposed = k_norm.transpose(-2, -1)
        print(f"k shape after transpose: {k_transposed.shape}")  # (B, N_h, C_h, H_1*W_1, N*N)

        attn_scores = q_norm @ k_transposed
        print(f"Attention scores shape (q @ k^T): {attn_scores.shape}")  # (B, N_h, C_h, N*N, N*N)

        attn = attn_scores * self.temperature
        print(f"Attention scores shape after temperature: {attn.shape}")  # (B, N_h, C_h, N*N, N*N)

        attn_softmax = attn.softmax(dim=-1)
        print(f"Attention map shape after softmax: {attn_softmax.shape}")  # (B, N_h, C_h, N*N, N*N)

        out_attention = attn_softmax @ v_rearranged
        print(f"Output shape after attn @ v: {out_attention.shape}")  # (B, N_h, C_h, N*N, H_1*W_1)

        out_rearranged = rearrange(
            out_attention,
            "b head c (N_h N_w) (h1 w1) -> b (head c) (h1 N_h) (w1 N_w)",
            head=self.num_heads,
            N_h=self.N,
            N_w=self.N,
            h1=h_after // self.N,
            w1=w_after // self.N,
        )
        print(f"Output shape after second rearrange: {out_rearranged.shape}")  # (B, C_in, H_after, W_after)

        out_proj = self.project_out(out_rearranged)
        print(f"Output shape after project_out: {out_proj.shape}")  # (B, C_in, H_after, W_after)

        # Remove padding
        out = out_proj[:, :, :h, :w]
        print(f"Final output shape after unpadding: {out.shape}")  # (B, C_in, H, W)
        print(f"--- End of forward pass ---")

        return out


# --- Test Configuration ---
batch_size = 2
dim = 32  # Must be divisible by num_heads
height = 47  # Not a multiple of blocks
width = 29  # Not a multiple of blocks
num_heads = 8  # Must divide dim
blocks = 8  # N

# --- Run the test ---
if __name__ == "__main__":
    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, dim, height, width)
    print(f"Original Input Tensor shape: {input_tensor.shape}")

    # Instantiate the module
    attention_module = CxNNxNN_Attention(dim, num_heads, bias=False, blocks=blocks)

    # Pass the input through the module
    output_tensor = attention_module(input_tensor)

    # Compare input and output shapes
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    assert input_tensor.shape == output_tensor.shape, "Input and output shapes do not match!"
    print("\nInput and output shapes match successfully.")
