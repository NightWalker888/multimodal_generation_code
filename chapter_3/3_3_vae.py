# 定义损失函数
def loss_function(reconstructed_data, original_data, mean, log_variance): 
    reconstruction_loss = mean_squared_error(reconstructed_data, original_data) 
    kl_loss = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()) 
    total_loss = reconstruction_loss + kl_loss
    return total_loss

# 定义优化器（如梯度下降优化器）
optimizer = optimizer.Adam(variational_autoencoder.parameters(), lr=learning_rate) 
# 开始训练循环
for epoch in range(epochs): 
    for batch in dataset_loader.get_batches(training_data, batch_size): 
        # 清零梯度
        optimizer.zero_grad() 
        
        # 将本批次数据传递给 VAE 
        mean, log_variance = variational_autoencoder.encode(batch) 
        
        # 重参数化技巧
        z = mean + torch.exp(log_variance * 0.5) * torch.randn_like(log_variance) 
        
        # 解码
        reconstructed_data = variational_autoencoder.decode(z) 
        
        # 计算损失
        loss = loss_function(reconstructed_data, batch, mean, log_variance) 
        
        # 反向传播
        loss.backward() 
        
        # 更新参数
        optimizer.step()
