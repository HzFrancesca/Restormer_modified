import torch
from thop import profile, clever_format
import os
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

import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


custom_ops = {
    HWxHW_Attention: HWxHW_Attention.HWxHW_macs,
    CxC_Attention: CxC_Attention.CxC_macs,
    CHxCH_Attention: CHxCH_Attention.CHxCH_macs,
    WxW_Attention: WxW_Attention.WxW_macs,
    CWxCW_Attention: CWxCW_Attention.CWxCW_macs,
    HxH_Attention: HxH_Attention.HxH_macs,
    CNNxCNN_Attention: CNNxCNN_Attention.CNNxCNN_macs,
    CxNNxNN_Attention: CxNNxNN_Attention.CxNNxNN_macs,
    CxHxH_Attention: CxHxH_Attention.CxHxH_macs,
    CxWxW_Attention: CxWxW_Attention.CxWxW_macs,
}


class Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CxWxW_Attention(8, 4, False)

    def forward(self, x):
        x = self.attn(x)
        return x


dummy_input = torch.randn(1, 8, 64, 96)  # (batch_size, channels, height, width)

# model = CxC_Attention(8, 4, False)
model = Custom()

# macs, params = profile(model, inputs=(dummy_input,), verbose=True)
macs, _ = profile(model, inputs=(dummy_input,), custom_ops=custom_ops, verbose=True)
params = sum(p.numel() for p in model.parameters())
macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

print(f"Model MACs: {macs_formatted}")
# print(f"Model macs (approx. 2*MACs): {clever_format([macs * 2, params], '%.3f')[0]}")
print(f"Model Params: {params_formatted}")
