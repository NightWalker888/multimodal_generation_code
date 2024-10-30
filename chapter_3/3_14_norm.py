# x的形状应为 [batch_size, channels, height, width]
def custom_batchnorm2d(x, gamma, beta, epsilon=1e-5):
    # 计算批量均值和方差
    mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
    var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)

    # 归一化
    x_normalized = (x - mean) / torch.sqrt(var + epsilon)
  
    # 伸缩偏移变换
    out = gamma * x_normalized + beta

    return out

def custom_groupnorm(x, gamma, beta, num_groups, epsilon=1e-5):
    N, C, H, W = x.size()
    G = num_groups
    # reshape the input tensor to shape: (N, G, C // G, H, W)
    x_grouped = x.reshape(N, G, C // G, H, W)

    # 计算组均值和方差
    mean = torch.mean(x_grouped, dim=(2, 3, 4), keepdim=True)
    var = torch.var(x_grouped, dim=(2, 3, 4), unbiased=False, keepdim=True)

    # 归一化
    x_grouped = (x_grouped - mean) / torch.sqrt(var + epsilon)
    
    # 伸缩偏移变换
    out = gamma * x_grouped + beta
    
    # reshape back to the original input shape
    out = out.reshape(N, C, H, W)
    
    return out

def custom_layernorm2d(x, gamma, beta, epsilon=1e-5):
    # 计算批量均值和方差
    mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
    var = torch.var(x, dim=(1, 2, 3), unbiased=False, keepdim=True)

    # 归一化
    x_normalized = (x - mean) / torch.sqrt(var + epsilon)
  
    # 伸缩偏移变换
    out = gamma * x_normalized + beta

    return out
    
def custom_instancenorm2d(x, gamma, beta, epsilon=1e-5):
    # 计算批量均值和方差
    mean = torch.mean(x, dim=(2, 3), keepdim=True)
    var = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)

    # 归一化
    x_normalized = (x - mean) / torch.sqrt(var + epsilon)
  
    # 伸缩偏移变换
    out = gamma * x_normalized + beta

    return out