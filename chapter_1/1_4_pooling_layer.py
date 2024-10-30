import torch
import torch.nn as nn

# 定义最大池化层
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 定义平均池化层
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# 创建一个随机的输入特征图
input = torch.randn(1, 1, 4, 4)  # 假设有一个单通道 4×4 的特征图

# 应用最大池化
output_max_pool = max_pool(input)

# 应用平均池化
output_avg_pool = avg_pool(input)

print("Input:\n", input)
print("Output after Max Pooling:\n", output_max_pool)
print("Output after Average Pooling:\n", output_avg_pool)
