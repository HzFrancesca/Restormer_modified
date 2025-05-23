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
from attention_OCA import OCA

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model = HWxHW_Attention(8, 4, False)
model = OCA(3, 4, False, 8, 0.5, 16)

input_size = (1, 3, 128, 128)  # (batch_size, channels, height, width)

summary(model, input_size=input_size)
