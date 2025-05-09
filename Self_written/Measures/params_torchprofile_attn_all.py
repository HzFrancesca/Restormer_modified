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
    print("错误：无法从 'attentions.py' 导入 Attention 模块。")
    print("请确保您有包含所需类的 'attentions.py' 文件，并且它在您的 Python 路径中。")
    exit()

from torchprofile import profile_macs
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_tensor = torch.randn(1, 8, 64, 96)

module_args = (8, 4, False)

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

print(f"开始计算 Attention 模块的 MACs 和参数数量...")
print(f"输入张量形状: {input_tensor.shape}")
print(f"模块初始化参数: {module_args}\n")


for AttentionClass in attention_classes:
    class_name = AttentionClass.__name__
    try:
        print(f"--- 处理 {class_name} ---")

        # 实例化模块，使用 * 解包 module_args 元组
        model = AttentionClass(*module_args)

        # 如果需要，可以将模型和输入移动到 GPU
        if torch.cuda.is_available():
            model = model.cuda()
            input_tensor_gpu = input_tensor.cuda()  # 使用单独的变量以防需要原始 CPU 输入

        # 计算 MACs
        # profile_macs 需要输入张量放在一个元组里
        macs = profile_macs(model, (input_tensor_gpu,))  # 如果使用了 GPU
        # macs = profile_macs(model, (input_tensor,))  # 在 CPU 上计算

        params = sum(p.numel() for p in model.parameters())

        print(f"  Macs: {macs}")  # 将 MACs 转换为 GigaMACs，保留4位小数
        print(f"  Params: {params}")

    except Exception as e:
        print(f"  错误：无法处理 {class_name}。原因：{e}")

    print("-" * (len(class_name) + 17))  # 打印分隔线

print("所有指定的 Attention 模块处理完成。")
