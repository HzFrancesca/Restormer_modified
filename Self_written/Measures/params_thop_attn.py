import torch
from thop import profile, clever_format
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


model = HWxHW_Attention(8, 4, False)
input_size = (1, 8, 64, 96)  # (batch_size, channels, height, width)
dummy_input = torch.randn(*input_size)


flops, params = profile(model, inputs=(dummy_input,), verbose=True)
flops, params = clever_format([flops, params], "%.3f")

print(f"Model FLOPs: {flops}")
print(f"Model Params: {params}")
