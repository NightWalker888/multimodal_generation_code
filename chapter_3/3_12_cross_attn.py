import torch 
import torch.nn as nn 
import torch.nn.functional as F 
class MultiHeadCrossAttention(nn.Module): 
    def __init__(self, num_heads, d_model, d_key, d_value): 
        super(MultiHeadCrossAttention, self).__init__() 
        self.num_heads = num_heads 
        self.d_key = d_key 
        self.d_value = d_value 
        self.W_q = nn.Linear(d_model, num_heads * d_key) 
        self.W_k = nn.Linear(d_model, num_heads * d_key) 
        self.W_v = nn.Linear(d_model, num_heads * d_value) 
        self.fc = nn.Linear(num_heads * d_value, d_model) 

    def forward(self, query, key, value): 
        batch_size = query.size(0) 
        # 线性映射得到 Q、K、V 
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_key) 
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_key) 
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_value) 
        # 计算注意力分数，缩放后进行归一化处理
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(self.d_key) 
        attn = F.softmax(scores, dim=-1) 
        context = torch.matmul(attn, V) 
        # 拼接多头注意力，然后经过线性映射得到输出特征
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, 
        self.num_heads * self.d_value) 
        output = self.fc(context) 
        return output 
    
# 假设参数
num_heads = 8 
d_model = 512 # 模型的维度
d_key = 64 # 矩阵和查询矩阵的维度
d_value = 64 # 值矩阵的维度
# 创建模型实例
cross_attention = MultiHeadCrossAttention(num_heads, d_model, d_key, d_value) 
# 示例文本特征和图像特征（需要通过模型（如 CLIP 和 U-Net）获取）
text_features = torch.rand(1, 10, d_model) # (batch_size, seq_len, d_model) 
image_features = torch.rand(1, 10, d_model) # 假设图像特征具有相同的形状
# 应用交叉注意力
# 在这里，图像特征作为查询矩阵，文本特征作为键矩阵和值矩阵
output = cross_attention(image_features, text_features, text_features)