import torch
from torchprofile import profile
import os

# 假设以下导入是有效的，并且你的 attentions.py 文件包含这些类
# 如果你运行此代码时遇到 ImportError，请确保 attentions.py 在正确的位置
# 并且这些类已正确定义。
try:
    from attentions import (
        HWxHW_Attention,
        CxC_Attention,
        CHxCH_Attention,
        WxW_Attention,
        CWxCW_Attention,
        HxH_Attention,
        CNNxCNN_Attention,
        CxNNxNN_Attention,
        CxHxH_Attention,
        CxWxW_Attention,
    )
except ImportError:
    print("Error: Could not import attention modules from 'attentions'.")
    print("Please ensure 'attentions.py' exists and contains the required classes.")
    # As a fallback for demonstration, let's define dummy classes.
    # **IMPORTANT**: These dummy classes will NOT give correct profiling results.
    # You MUST have your actual attention implementations for meaningful numbers.
    import torch.nn as nn

    class DummyAttention(nn.Module):
        def __init__(self, dim, num_heads, bias):
            super().__init__()
            # Add a dummy layer and parameter to make it profileable, but results won't reflect real attention
            self.dummy_linear = nn.Linear(dim, dim)
            self.dummy_param = nn.Parameter(torch.randn(1))

        def forward(self, x):
            # Dummy forward: simply return input modified slightly
            # The real attention forward pass is crucial for correct profiling
            return self.dummy_linear(x.transpose(1, 3)).transpose(1, 3)

    print("Using DummyAttention classes for demonstration. Results will not be accurate.")
    HWxHW_Attention = DummyAttention
    CxC_Attention = DummyAttention
    CHxCH_Attention = DummyAttention
    WxW_Attention = DummyAttention
    CWxCW_Attention = DummyAttention
    HxH_Attention = DummyAttention
    CNNxCNN_Attention = DummyAttention
    CxNNxNN_Attention = DummyAttention
    CxHxH_Attention = DummyAttention
    CxWxW_Attention = DummyAttention


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置 CUDA 设备，尽管 torchprofile 通常在 CPU 上执行计算

# 要进行分析的注意力模块类列表
attention_classes = [
    HWxHW_Attention,
    CxC_Attention,
    CHxCH_Attention,
    WxW_Attention,
    CWxCW_Attention,
    HxH_Attention,
    CNNxCNN_Attention,
    CxNNxNN_Attention,
    CxHxH_Attention,
    CxWxW_Attention,
]

# 实例化模块的参数
instantiation_params = (8, 4, False)  # (dim, num_heads, bias) 根据你的模块定义可能需要调整

# Dummy 输入张量 (batch_size, channels, height, width)
# 请确保通道数 (8) 与实例化参数中的 dim (8) 匹配
dummy_input_size = (1, 8, 64, 96)
dummy_input = torch.randn(*dummy_input_size)

print("开始分析注意力模块的参数和 MACs:")
print("-" * 40)

for AttentionClass in attention_classes:
    model_name = AttentionClass.__name__
    print(f"正在分析: {model_name}")

    try:
        # 实例化模型
        model = AttentionClass(*instantiation_params)

        # 进行性能分析 (MACs 和 参数)
        # profile 函数需要模型和一个包含输入张量的元组
        # 注意：torchprofile 报告的是 MACs，通常 1 MACs ≈ 2 FLOPs
        # 但这里的术语使用 torchprofile 的输出 MACs
        macs, params = profile(model, inputs=(dummy_input,))

        # 打印结果
        # 将 MACs 和 Params 转换为百万 (M) 单位，保留小数点后 4 位
        print(f"  参数 (M): {params / 1e6:.4f}")
        print(f"  MACs (G): {macs / 1e9:.4f}")  # 通常 MACs 比较大，用 G (十亿) 单位
        print("-" * 40)

    except Exception as e:
        print(f"  分析 {model_name} 时出错: {e}")
        print("-" * 40)

print("分析完成。")
