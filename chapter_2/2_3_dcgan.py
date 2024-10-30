import torch 
import torch.nn as nn 

# DCGAN的生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入是一个Z维的噪声，将它映射成一个维度为1024的特征图
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # 上一步的输出形状：(1024, 4, 4)
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 上一步的输出形状：(512, 8, 8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 上一步的输出形状：(256, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 上一步的输出形状：(128, 32, 32)
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出形状：(1, 64, 64)
        )
    def forward(self, input):
        return self.main(input)


# DCGAN的判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入形状：(1, 64, 64)
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出形状：(128, 32, 32)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出形状：(256, 16, 16)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出形状：(512, 8, 8)
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出形状：(1024, 4, 4)
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


if __name__ == "__main__":

    generator = Generator()
    discriminator  = Discriminator()

    # 创建一个随机的输入特征图
    z = 100
    input = torch.randn(1, 100, 1, 1)  # 假设有一个通道数为100的 1×1 的特征图

    generator_result = generator(input)
    discriminator_result = discriminator(generator_result)

    print(generator_result.shape, discriminator_result.shape)