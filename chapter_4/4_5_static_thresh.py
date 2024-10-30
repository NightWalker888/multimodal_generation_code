import numpy as np 

def apply_static_threshold(noise): 
    # 将噪声限制在−1 到 1 之间
    noise_clipped = np.clip(noise, −1, 1) 
    return noise_clipped 

# 示例
noise = np.random.randn(100, 100) # 假设这是 U-Net 模型预测的噪声
noise_after_static_threshold = apply_static_threshold(noise)