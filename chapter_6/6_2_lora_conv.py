import torch 
import torch.nn as nn 
class LoRA_Conv2d(nn.Module): 
 def __init__(self, in_channels, out_channels, kernel_size, r, stride=1, 
 padding=0, dilation=1, groups=1, bias=True): 
    super(LoRA_Conv2d, self).__init__() 
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
    padding, dilation, groups, bias)
    self.A = nn.Parameter(torch.randn(r * kernel_size, in_channels * 
    kernel_size)) 
    self.B = nn.Parameter(torch.randn(out_channels//groups * kernel_size, 
    r * kernel_size)) 
    # 冻结 self.conv 中的所有参数
    for param in self.conv.parameters(): 
        param.requires_grad = False 

    def forward(self, x): 
        delta_W = (self.B @ self.A).view(self.conv.weight.shape) 
        self.conv.weight.data += delta_W 
        return self.conv(x) 
    
# 示例：输入通道数为 16，输出通道数为 32，卷积核尺寸为 3×3，r 为 5 
in_channels = 16 
out_channels = 32 
kernel_size = 3 
r = 5 
lora_conv = LoRA_Conv2d(in_channels, out_channels, kernel_size, r, groups = 2) 
x = torch.randn(1,16,64,64) # 输入特征
y = lora_conv(x) # 输出特征
print(y.shape)