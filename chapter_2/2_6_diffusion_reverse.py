import torch 

for t_step in reversed(range(T)): # 从 T 开始向 0 迭代
    t = t_step 
    t = torch.tensor(t).to(device) 
    
    # 如果时间步大于 0，则随机生成一个高斯噪声
    # 如果时间步为 0，即已经回到原始图像，则无须再添加噪声
    z = torch.randn_like(x_t,device=device) if t_step > 0 else 0 
    
    # 使用 DDPM 采样器并根据式（2.4）进行计算（此步骤中额外添加了一个高斯噪声）
    x_t_minus_one = torch.sqrt(1/alphas[t])*(x_t-(1-alphas[t]) \ 
    *model(x_t,t.reshape(1,))/torch.sqrt(1-alphas_bar[t])) \ 
    +torch.sqrt(betas[t])*z 
    
    x_t = x_t_minus_one