import torch 
import torch.nn as nn 
import torch.optim as optim 

# 假设生成器、判别器, data_loader, device, num_epochs 和 latent_dim 已经定义

# 生成器和判别器的优化器
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) 
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 
0.999)) 

# 损失函数
adversarial_loss = nn.BCELoss() 

for epoch in range(num_epochs): 
    for real_batch in data_loader: 
 
        # 更新判别器
        real_images = real_batch.to(device) 
        batch_size = real_images.size(0) 
        
        # 生成图像
        noise = torch.randn(batch_size, latent_dim, device=device) 
        fake_images = generator(noise) 
        
        # 判别器在真实图像上的损失
        real_labels = torch.ones(batch_size, 1, device=device) 
        fake_labels = torch.zeros(batch_size, 1, device=device) 
        
        disc_real_loss = adversarial_loss(discriminator(real_images), 
        real_labels) 
        disc_fake_loss = adversarial_loss(discriminator(fake_images.detach()), 
        fake_labels) 
        
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

print(f"Epoch [{epoch+1}/{num_epochs}], Disc Loss: {disc_loss.item()}, Gen 
Loss: {gen_loss.item()}")