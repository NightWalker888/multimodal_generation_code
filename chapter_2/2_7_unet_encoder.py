import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # 第一次卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 第二次卷积层，通常使用相同的out_channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 应用第一次卷积
        x = self.conv1(x)
        # 应用第二次卷积
        x = self.conv2(x)
        return x
    

class DownSample(nn.Module): 
    """下采样层""" 
    def __init__(self, in_channels, out_channels): 
        super().__init__() 
        self.pooling_layer = nn.Sequential( 
        nn.MaxPool2d(2), 
        DoubleConv(in_channels, out_channels) 
        ) 

    def forward(self, x): 
        return self.pooling_layer(x)
    

class UNetEncoder(nn.Module): 
    def __init__(self, input_channels): 
        super(UNetEncoder, self).__init__() 
        self.input_channels = input_channels 
        self.entry_conv = DoubleConv(self.input_channels, 64) 
        self.down1 = DownSample(64, 128) 
        self.down2 = DownSample(128, 256) 
        self.down3 = DownSample(256, 512) 

    def forward(self, input_tensor): 
        # 入口层
        feature1 = self.entry_conv(input_tensor) 
        # 连续下采样
        feature2 = self.down1(feature1) 
        feature3 = self.down2(feature2) 
        feature4 = self.down3(feature3) 
        return feature4


# 使用示例
# 假设输入特征图的通道数为3
unet_encoder = UNetEncoder(input_channels=3)
sampler_input = torch.randn(1,3,256,256) # 假设输入是一个batch size为1的256x256图像
output = unet_encoder(sampler_input)
print(output.shape) # 输出特征图的尺寸