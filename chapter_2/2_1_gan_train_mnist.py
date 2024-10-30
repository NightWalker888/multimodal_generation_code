import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image 


# 为了可视化生成的图像，我们可以定义一个辅助函数
def show_generated_images(generator, num_images=25, latent_dim=100, device='cpu'):
    generator.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        noise = torch.randn(num_images, latent_dim, device=device)
        fake_images = generator(noise)
        fake_images = fake_images.reshape(fake_images.shape[0], 1, 28, 28)  # 重塑为图像的形状
        fake_images = fake_images.to('cpu')  # 移动到 CPU
        fake_images = fake_images.detach().squeeze()  # 转换为 numpy 数组并移除单维度条目

    # 将生成的图像转换为 PIL 图像并返回
    pil_images = []
    for img in fake_images:
        img = img.numpy()  # 转换为 numpy 数组
        img = img.reshape(28, 28)  # 重塑为 28x28
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0, 1]
        img = (255 * img).astype('uint8')  # 转换为 [0, 255]
        pil_img = Image.fromarray(img, 'L')  # 创建 PIL 图像
        pil_images.append(pil_img)

    generator.train()  # 恢复训练模式
    return pil_images


def create_image_grid(images, grid_size):
    if len(images) != grid_size[0] * grid_size[1]:
        raise ValueError("Number of images must match the grid size.")

    # 创建一个新的图像，背景为白色
    grid_image = Image.new('L', (grid_size[1] * 28, grid_size[0] * 28), 'white')

    for i, image in enumerate(images):
        # 计算图像的粘贴位置
        x = (i % grid_size[1]) * 28
        y = (i // grid_size[1]) * 28
        grid_image.paste(image, (x, y))

    return grid_image


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

# 数据集的转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 假设生成器、判别器, device, num_epochs 和 latent_dim 已经定义
# 这里device设置为'cpu'
device = torch.device('cpu')


latent_dim = 100 # 生成器输入的维度
hidden_dim = 256 # 隐藏层维度
output_dim = 784 # 生成器输出的维度，例如对于 28px×28px 的图像，输出的维度是 784 
generator = Generator(latent_dim, hidden_dim, output_dim) 
discriminator = Discriminator(output_dim, hidden_dim)
num_epochs = 50


# 生成器和判别器的优化器
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 损失函数
adversarial_loss = nn.BCELoss()

for epoch in range(num_epochs):
    for real_batch in train_loader:
        # 更新判别器
        real_images = real_batch[0].reshape(real_batch[0].shape[0], -1).to(device)  # real_batch[0] 是图片，real_batch[1] 是标签
        batch_size = real_images.size(0)

        # 生成图像
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(noise)

        # 判别器在真实图像上的损失
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        disc_real_loss = adversarial_loss(discriminator(real_images), real_labels)
        disc_fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake_labels)

        disc_loss = disc_real_loss + disc_fake_loss

        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # 更新生成器
        noise = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(noise)

        gen_loss = adversarial_loss(discriminator(fake_images), real_labels)

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()


    # 可视化中间结果
    pil_images = show_generated_images(generator, num_images=25, latent_dim=latent_dim, device='cpu')

    # 将图像列表转换为网格形式（例如，5x5 网格）
    grid_size = (5, 5)
    grid_image = create_image_grid(pil_images, grid_size)

    # 保存拼接后的图像
    grid_image.save(f'generated_images_{epoch+1}.png')

    print(f"Epoch [{epoch+1}/{num_epochs}], Disc Loss: {disc_loss.item()}, Gen Loss: {gen_loss.item()}")