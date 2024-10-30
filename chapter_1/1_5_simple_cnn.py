import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 第一个卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       # 池化层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 第二个卷积层
        self.fc1 = nn.Linear(32 * 14 * 14, 3)                              # 全连接层
        self.fc2 = nn.Linear(3, 2)                                         # 全连接层

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积 -> 激活 -> 池化
        x = self.pool(F.relu(self.conv2(x)))  # 卷积 -> 激活 -> 池化
        x = torch.flatten(x, 1)               # 展平
        x = torch.relu(self.fc1(x))            # 全连接
        x = torch.softmax(self.fc2(x), dim=1)        # 全连接
        return x

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor()           # 转换为tensor
    ])
    image = Image.open(image_path)
    image = transform(image).float()
    image = image.unsqueeze(0)  # 添加一个批量维度
    return image

# 图像路径，例如：'path/to/your/image.jpg'
image_path = 'test.png'

# 加载和处理图像
image = process_image(image_path)
print(image.shape)

# 创建模型实例并进行预测
model = SimpleCNN()
output = model(image)

print("Predicted Value:", output)

