from torchprofile import profile_macs
import torch
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

model = CxC_Attention(8, 4, False)

macs = profile_macs(model, input)
params = sum(p.numel() for p in model.parameters())
print(f"  Macs: {macs}")
print(f"  Params: {params} ")
