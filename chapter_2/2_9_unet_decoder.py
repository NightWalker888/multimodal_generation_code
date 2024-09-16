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

class Upsample(nn.Module): 
    """上采样层""" 
    def __init__(self, in_channels, out_channels): 
        super(Upsample, self).__init__() 
        self.up_conv = nn.ConvTranspose2d(in_channels , in_channels // 2, 
        kernel_size=2, stride=2) 
        self.post_conv = DoubleConv(in_channels, out_channels) 

    def forward(self, x, skip_x): 
        x = self.up_conv(x) 
        x = torch.cat([skip_x, x], dim=1) 
        return self.post_conv(x) 
 
class FinalConv(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(FinalConv, self).__init__() 
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x): 
        return self.final_conv(x) 
    
class UNetDecoder(nn.Module): 
    def __init__(self, n_classes): 
        super(UNetDecoder, self).__init__() 
        self.n_classes = n_classes 
        self.up1 = Upsample(512, 256) 
        self.up2 = Upsample(256, 128) 
        self.up3 = Upsample(128, 64) 
        self.final_conv = FinalConv(64, n_classes) 

    def forward(self, x4, x3, x2, x1): 
        x = self.up1(x4, x3)
        x = self.up2(x, x2) 
        x = self.up3(x, x1) 
        output = self.final_conv(x) 
        return output



# 使用示例
# 假设输入特征图的通道数为3
unet_decoder = UNetDecoder(n_classes=512)
sampler_input_4 = torch.randn(1,512,32,32) # 假设featuremap的维度为1*512*32*32
sampler_input_3 = torch.randn(1,256,64,64) # 假设featuremap的维度为1*256*64*64
sampler_input_2 = torch.randn(1,128,128,128) # 假设featuremap的维度为1*128*128*128
sampler_input_1 = torch.randn(1,64,256,256) # 假设featuremap的维度为1*64*256*256
output = unet_decoder(sampler_input_4, sampler_input_3, sampler_input_2, sampler_input_1)
print(output.shape) # 输出特征图的尺寸