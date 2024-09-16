for epoch in range(epochs): 
    for batch in dataset_loader.get_batches(training_data, batch_size): 
        # 清零梯度
        optimizer.zero_grad() 
        
        # 将本批次数据传递给 AE 
        encoded_data = autoencoder.encode(batch) 
        reconstructed_data = autoencoder.decode(encoded_data) 
        
        # 计算损失，例如使用 L2 损失
        loss = loss_function(reconstructed_data, batch) 
        
        # 反向传播
        loss.backward() 
        
        # 更新参数
        optimizer.step()