# 加入噪声函数
def add_noise(data, factor): 
    noise = factor * np.random.normal(size=data.shape) 
    noisy_data = data + noise 
    return noisy_data.clip(0, 1)


# 开始训练循环
for epoch in range(epochs): 
    for batch in dataset_loader.get_batches(training_data, batch_size): 
    
        # 给本批次数据加入噪声
        noisy_batch = add_noise(batch, noise_factor) 
        
        # 清零梯度
        optimizer.zero_grad() 
        
        # 将带噪声的本批次数据传递给 DAE 
        encoded_data = denoising_autoencoder.encode(noisy_batch) 
        reconstructed_data = denoising_autoencoder.decode(encoded_data) 
        
        # 计算损失
        loss = loss_function(reconstructed_data, batch) 
        
        # 反向传播
        loss.backward() 
        
        # 更新参数
        optimizer.step()