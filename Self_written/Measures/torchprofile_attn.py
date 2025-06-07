import torch
from torchprofile_restormer_swin import WindowAttention
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
from torchprofile import profile_macs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dummy_input = torch.randn(1, 8, 64, 96).to(device)
model = CxC_Attention(8, 4, False).to(device)

# profile_macs 需要输入张量放在一个元组里
macs = profile_macs(model, (dummy_input,))
params = sum(p.numel() for p in model.parameters())

print(f"  Macs: {macs}  ")
print(f"  Params: {params}")
