import torch 
import torch.nn as nn 

# 创建一个转置卷积层实例
# 假设输入特征图的通道数为 128，输出特征图的通道数为 64 
transpose_conv = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) 
# 创建一个假设的输入特征图
# 假设批次大小为 1，通道数为 128，高度和宽度为 32px×32px 
input_tensor = torch.randn(1, 128, 32, 32) 
# 通过转置卷积层进行上采样
output_tensor = transpose_conv(input_tensor) 
# 输出结果的尺寸
print("Output Tensor Shape:", output_tensor.shape) 
# 输出结果为：Output Tensor Shape: torch.Size([1, 64, 64, 64])