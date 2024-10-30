import torch 
from denoising_diffusion_pytorch import UNet, GaussianDiffusion, Trainer 

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

trainer = Trainer( 
 diffusion, 
 './oxford-datasets/raw-images', 
 train_batch_size = 16, 
 train_lr = 2e-5, 
 train_num_steps = 30000, # 总共训练 30000 步
 gradient_accumulate_every = 2, # 梯度累积步数
 ema_decay = 0.995, # 指数滑动平均衰退参数
 amp = True, # 使用混合精度训练加速
 calculate_fid = False, # 关闭 FID 评测指标计算，FID 用于评测生成质量
 save_and_sample_every = 5000 # 每隔 5000 步保存一次模型权重
) 
trainer.train()