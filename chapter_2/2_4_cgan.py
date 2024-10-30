import torch 
import torch.nn as nn 

# cGAN 的生成器和判别器架构与 DCGAN 的类似，但输入包含额外的条件信息（如标签信息）
class ConditionalGenerator(nn.Module): 
    def __init__(self, condition_dim): 
        super(ConditionalGenerator, self).__init__() 
        # 假设 condition_dim 是条件向量的维度
        self.condition_dim = condition_dim 
        self.main = nn.Sequential(
            # 输入是一个Z维的噪声，将它映射成一个维度为1024的特征图
            nn.ConvTranspose2d(in_channels=100 + self.condition_dim, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, noise, condition): 
        # 假设 condition 是条件向量
        condition = condition.view(-1, self.condition_dim, 1, 1) 
        input = torch.cat([noise, condition], 1) # 在通道维度上合并噪声和条件向量
        return self.main(input)
    
class ConditionalDiscriminator(nn.Module): 
    def __init__(self): 
        super(ConditionalDiscriminator, self).__init__() 
        # 其他层的定义保持不变
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

    generator = ConditionalGenerator(condition_dim = 10)
    discriminator  = ConditionalDiscriminator()

    # 创建一个随机的输入特征图
    z = 100
    input = torch.randn(1, 100, 1, 1)  # 假设有一个通道数为100的 1×1 的特征图
    condition = torch.randn(1, 10, 1, 1)  # 假设有一个通道数为10的 1×1 的condition

    generator_result = generator(input, condition)
    discriminator_result = discriminator(generator_result)

    print(generator_result.shape, discriminator_result.shape)