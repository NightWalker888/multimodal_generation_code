import torch 
import torch.nn as nn 
import numpy as np 
import cv2 

class Net(nn.Module):
    """定义一个人工神经网络，只包含一个卷积操作
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    # 随机初始化一张图像输入
    input = torch.rand(1,3,256,256)
    net = Net()

    # 获取中间的特征图
    feature_maps = net(input)
    print(feature_maps.shape)
