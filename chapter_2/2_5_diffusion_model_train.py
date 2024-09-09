import torch 

for i, (x_0) in enumerate(tqdm_data_loader): 
    # 将数据加载至相应的运行设备(device) 
    x_0 = x_0.to(device) 
    
    # 对于每张图像，随机在 1～T 的时间步中进行采样
    t = torch.randint(1, T, size=(x_0.shape[0],), device=device) 
    
    # 取得各时间步 t 对应的 alpha_t 的开方结果的连乘
    sqrt_alpha_t_bar = torch.gather(sqrt_alphas_bar, dim=0, 
    index=t).reshape(-1, 1, 1, 1) 
    
    # 取得各时间步 t 对应的 1-alpha_t 的开方结果的连乘
    sqrt_one_minus_alpha_t_bar = torch.gather(sqrt_one_minus_alphas_bar, 
    dim=0, index=t).reshape(-1, 1, 1, 1)

    # 随机生成一个高斯噪声
    noise = torch.randn_like(x_0).to(device) 
    
    # 计算第 t 步的加噪图像 x_t 
    x_t = sqrt_alpha_t_bar * x_0 + sqrt_one_minus_alpha_t_bar * noise 
    
    # 将 x_t 输入 U-Net 模型，得到预测的噪声
    out = net_model(x_t, t) 
    
    loss = loss_function(out, noise) # 用预测的噪声和随机生成的高斯噪声计算损失
    optimizer.zero_grad() # 将优化器的梯度清零
    loss.backward() # 对损失函数反向求导以计算梯度
    optimizer.step() # 更新优化器参数