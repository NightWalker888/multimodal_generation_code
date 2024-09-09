import torch 

# cGAN 的生成器和判别器架构与 DCGAN 的类似，但输入包含额外的条件信息（如标签信息）
class ConditionalGenerator(nn.Module): 
    def __init__(self, condition_dim): 
        super(ConditionalGenerator, self).__init__() 
        # 假设 condition_dim 是条件向量的维度
        self.condition_dim = condition_dim 
        # 其他层的定义保持不变

    def forward(self, noise, condition): 
        # 假设 condition 是条件向量
        condition = condition.view(-1, self.condition_dim, 1, 1) 
        input = torch.cat([noise, condition], 1) # 在通道维度上合并噪声和条件向量
        return self.main(input) 
    
class ConditionalGenerator(nn.Module): 
    def __init__(self, condition_dim): 
        super(ConditionalGenerator, self).__init__() 
        # 假设 condition_dim 是条件向量的维度
        self.condition_dim = condition_dim 
        # 其他层的定义保持不变

    def forward(self, noise, condition): 
        # 假设 condition 是条件向量
        condition = condition.view(-1, self.condition_dim, 1, 1) 
        input = torch.cat([noise, condition], 1) # 在通道维度上合并噪声和条件向量
        return self.main(input)