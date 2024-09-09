import torch
from denoising_diffusion_PyTorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000   # 加噪总步数
).cuda()

trainer = Trainer(
    diffusion,
    './oxford-datasets/raw-images',
    train_batch_size = 16,
    train_lr = 2e-5,
    train_num_steps = 30000,          # 总共训练30000步
    gradient_accumulate_every = 2,    # 梯度累积步数
    ema_decay = 0.995,                # 指数滑动平均decay参数
    amp = True,                       # 使用混合精度训练加速
    calculate_fid = False,            # 我们关闭FID评测指标计算（比较耗时）。FID用于评测生成质量。
    save_and_sample_every = 5000      # 每隔2000步保存一次模型
)

trainer.train()