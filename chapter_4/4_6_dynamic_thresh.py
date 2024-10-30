import numpy as np 
def apply_dynamic_threshold(noise, percentile=90): 
    # 计算动态阈值
    s = np.percentile(np.abs(noise), percentile) 
    
    # 将噪声限制在−s 到 s 之间
    noise_clipped = np.clip(noise, −s, s) 
    
    # 标准化噪声，使其在−1 到 1 的范围内
    noise_normalized = noise_clipped / s 
    return noise_normalized 
# 示例
noise = np.random.randn(100, 100) # 假设这是 U-Net 模型预测的噪声
noise_after_dynamic_threshold = apply_dynamic_threshold(noise)