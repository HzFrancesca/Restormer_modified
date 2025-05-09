import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


input = torch.randn(1, 8, 64, 96)  # (batch_size, channels, height, width)

model = HWxHW_Attention(8, 4, False)


print(f"正在分析模型: {model.__class__.__name__}")
print(f"输入形状: {input.shape}")
print("-" * 30)

# 1. 使用 fvcore 计算 FLOPs (浮点运算量)
# FlopCountAnalysis 需要模型实例和 dummy 输入
flops_analyzer = FlopCountAnalysis(model, input)

# 获取总的 FLOPs
total_flops = flops_analyzer.total()
print(f"总 FLOPs: {total_flops} ({total_flops / 1e9:.4f} GFLOPs)")  # 通常以 GFLOPs 显示更直观


# 2. 使用 fvcore 计算参数量 (Parameters)
# parameter_count 函数只需要模型实例
total_params = parameter_count(model)
print(f"总参数量: {total_params} ({total_params / 1e6:.4f} Millions)")  # 通常以 Millions 或 Billions 显示


# 3. 可选：按模块查看 FLOPs
# 这可以帮助你了解模型中哪个部分计算量最大
print("\n按模块查看 FLOPs:")
flops_by_module = flops_analyzer.by_module()
# 按 FLOPs 从大到小排序打印
sorted_flops_by_module = sorted(flops_by_module.items(), key=lambda item: item[1], reverse=True)
for module_name, flops in sorted_flops_by_module:
    print(f"  {module_name}: {flops}")

# 你也可以使用 flops_analyzer.by_module_and_operator() 查看更详细的分析
# print("\n按模块和操作符查看 FLOPs:")
# print(flops_analyzer.by_module_and_operator())

print("-" * 30)
