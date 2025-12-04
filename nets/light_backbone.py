import torch
import torch.nn as nn
from .darknet import BaseConv, DWConv, CSPLayer, Bottleneck


class LightBackbone(nn.Module):
    def __init__(self, out_features=("dark3", "dark4", "dark5"), depth=0.33, width=0.50, depthwise=True, act="silu"):
        super(LightBackbone, self).__init__()
        assert out_features, "please provide output features"
        self.out_features = out_features
        
        Conv = DWConv if depthwise else BaseConv
        
        # 大幅缩减通道数，只保证空间尺度相同
        # 通道数设计：dark3=32, dark4=64, dark5=128
        self.out_channels = {
            "dark3": 32,   # 对应CSPDarknet的dark3空间尺度 (H/8, W/8)
            "dark4": 64,   # 对应CSPDarknet的dark4空间尺度 (H/16, W/16)
            "dark5": 128,  # 对应CSPDarknet的dark5空间尺度 (H/32, W/32)
        }
        
        init_channels = 8  # 初始通道数
        
        self.stem = BaseConv(1, init_channels, ksize=3, stride=2, act=act)  # 下采样2倍: (B, 1, H, W) -> (B, 8, H/2, W/2)
        
        # Dark2层：轻量级特征提取
        self.dark2 = nn.Sequential(
            Conv(init_channels, init_channels * 2, 3, 2, act=act),  # 下采样2倍: (B, 8, H/2, W/2) -> (B, 16, H/4, W/4)
            CSPLayer(init_channels * 2, init_channels * 2, n=1, depthwise=depthwise, act=act),  # (B, 16, H/4, W/4)
        )
        
        # Dark3层：对应CSPDarknet的dark3空间尺度 (H/8, W/8)
        self.dark3 = nn.Sequential(
            Conv(init_channels * 2, self.out_channels["dark3"], 3, 2, act=act),  # 下采样2倍: (B, 16, H/4, W/4) -> (B, 32, H/8, W/8)
            CSPLayer(self.out_channels["dark3"], self.out_channels["dark3"], n=1, depthwise=depthwise, act=act),  # (B, 32, H/8, W/8)
        )
        
        # Dark4层：对应CSPDarknet的dark4空间尺度 (H/16, W/16)
        self.dark4 = nn.Sequential(
            Conv(self.out_channels["dark3"], self.out_channels["dark4"], 3, 2, act=act),  # 下采样2倍: (B, 32, H/8, W/8) -> (B, 64, H/16, W/16)
            CSPLayer(self.out_channels["dark4"], self.out_channels["dark4"], n=1, depthwise=depthwise, act=act),  # (B, 64, H/16, W/16)
        )
        
        # Dark5层：对应CSPDarknet的dark5空间尺度 (H/32, W/32)
        self.dark5 = nn.Sequential(
            Conv(self.out_channels["dark4"], self.out_channels["dark5"], 3, 2, act=act),  # 下采样2倍: (B, 64, H/16, W/16) -> (B, 128, H/32, W/32)
            CSPLayer(self.out_channels["dark5"], self.out_channels["dark5"], n=1, depthwise=depthwise, act=act, shortcut=False),  # (B, 128, H/32, W/32)
        )
    
    def forward(self, x):
        outputs = {}
        x = self.stem(x)  # (B, 8, H/2, W/2)

        # Dark2
        x = self.dark2(x)  # (B, 16, H/4, W/4)
        
        # Dark3: 空间尺度 H/8, W/8
        x = self.dark3(x)  # (B, 32, H/8, W/8)
        outputs["dark3"] = x
        
        # Dark4: 空间尺度 H/16, W/16
        x = self.dark4(x)  # (B, 64, H/16, W/16)
        outputs["dark4"] = x
        
        # Dark5: 空间尺度 H/32, W/32
        x = self.dark5(x)  # (B, 128, H/32, W/32)
        outputs["dark5"] = x
        
        return {k: v for k, v in outputs.items() if k in self.out_features}
