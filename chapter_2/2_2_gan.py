import torch 
import torch.nn as nn 

# 最初的 GAN 的生成器
class Generator(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim): 
        super(Generator, self).__init__() 
        self.net = nn.Sequential( 
        nn.Linear(input_dim, hidden_dim), 
        nn.ReLU(True), 
        nn.Linear(hidden_dim, hidden_dim), 
        nn.ReLU(True), 
        nn.Linear(hidden_dim, output_dim), 
        nn.Tanh() 
        ) 

    def forward(self, z): 
        return self.net(z) 
    
# 最初的 GAN 的判别器
class Discriminator(nn.Module): 
    def __init__(self, input_dim, hidden_dim): 
        super(Discriminator, self).__init__() 
        self.net = nn.Sequential( 
        nn.Linear(input_dim, hidden_dim), 
        nn.LeakyReLU(0.2, inplace=True), 
        nn.Linear(hidden_dim, hidden_dim), 
        nn.LeakyReLU(0.2, inplace=True), 
        nn.Linear(hidden_dim, 1), 
        nn.Sigmoid() 
        ) 
    def forward(self, x): 
        return self.net(x) 
 
if __name__ == "__main__": 
    # 初始化
    input_dim = 100 # 生成器输入的维度
    hidden_dim = 256 # 隐藏层维度
    output_dim = 784 # 生成器输出的维度，例如对于 28px×28px 的图像，输出的维度是 784 
    G = Generator(input_dim, hidden_dim, output_dim) 
    D = Discriminator(output_dim, hidden_dim)

    # 创建一个随机的输入特征图
    input = torch.randn(1, input_dim)  

    generator_result = G(input)
    discriminator_result = D(generator_result)

    print(generator_result.shape, discriminator_result.shape)