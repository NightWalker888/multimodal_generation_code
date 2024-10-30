import torch 
import torch.nn as nn 

class LoRA_FC(nn.Module): 
 def __init__(self, d, r): 
    super(LoRA_FC, self).__init__() 
    self.d = d 
    self.r = r 
    self.A = nn.Parameter(torch.randn(r, d)) 
    self.B = nn.Parameter(torch.randn(d, r)) 
    
    # 对于预训练模型，self.W 为预训练权重，不需要进行梯度更新
    self.W = nn.Parameter(torch.randn(d, d), requires_grad=False) 
 def forward(self, x): 
    delta_W = self.B @ self.A # 计算增量权重
    return (self.W + delta_W) @ x 
 
# 示例：d = 10000，r = 100 
d = 10000 
r = 100 
lora_fc = LoRA_FC(d, r) 
x = torch.randn(10000,1) # 输入特征
y = lora_fc(x) # 输出特征
print(y.shape)