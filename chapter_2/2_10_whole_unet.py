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

class FullUNet(nn.Module): 
    def __init__(self, input_channels, num_classes): 
        super(FullUNet, self).__init__() 
        self.input_channels = input_channels 
        self.num_classes = num_classes 
        self.encoder1 = DoubleConv(input_channels, 64) 
        self.encoder2 = DownSample(64, 128) 
        self.encoder3 = DownSample(128, 256) 
        self.encoder4 = DownSample(256, 512) 
        self.decoder1 = Upsample(512, 256) 
        self.decoder2 = Upsample(256, 128) 
        self.decoder3 = Upsample(128, 64) 
        self.classifier = FinalConv(64, num_classes) 


    def forward(self, input_tensor): 
        # 编码器部分 
        enc1 = self.encoder1(input_tensor) 
        enc2 = self.encoder2(enc1) 
        enc3 = self.encoder3(enc2) 
        enc4 = self.encoder4(enc3) 
        # 解码器部分，需要传入编码器的输出作为跳跃连接 
        dec1 = self.decoder1(enc4, enc3) 
        dec2 = self.decoder2(dec1, enc2) 
        dec3 = self.decoder3(dec2, enc1) 
        # 分类层 
        output = self.classifier(dec3) 
        return output 


# 使用示例 
unet_model = FullUNet(input_channels=3, num_classes=2) 
sample_input = torch.randn(1, 3, 256, 256) 
# 假设输入是一个 batch size 为 1 的 256x256 图像 
output = unet_model(sample_input) 
print(output.shape)