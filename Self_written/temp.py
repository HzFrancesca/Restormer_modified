# 示例代码验证
import torch
import einops

# 假设输入张量形状为 (batch, num_heads * head_dim, h, w)
q = torch.randn(2, 6, 4, 5)  # batch=2, head=2, head_dim=3, h=4, w=5
num_heads = 2
c = 3  # head_dim

# 执行重排
q_rearranged = einops.rearrange(q, "b (head c) h w -> b head w (c h)", head=num_heads)

# 输出形状应为 (2, 2, 5, 12)
print(q_rearranged.shape)  # torch.Size([2, 2, 5, 12])
