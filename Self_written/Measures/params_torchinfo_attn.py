from torchinfo import summary
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

summary(model, input_size=input_size)
