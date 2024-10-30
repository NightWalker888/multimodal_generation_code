from denoising_diffusion_pytorch import UNet, GaussianDiffusion 
import torch 

# 实例化一个 U-Net 模型，设置基础维度和不同层级的维度倍数
model = UNet( 
 dim = 64, 
 dim_mults = (1, 2, 4, 8) 
).cuda() 

# 实例化一个高斯扩散模型，配置其底层使用的 U-Net 模型、图像尺寸和总加噪步数
diffusion = GaussianDiffusion( 
 model, 
 image_size = 128, 
 timesteps = 1000 # 总加噪步数
).cuda()

# 使用随机初始化的图像进行一次训练
training_images = torch.randn(8, 3, 128, 128) 
loss = diffusion(training_images.cuda()) 
loss.backward()