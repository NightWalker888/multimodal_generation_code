# 导入所需的库
import torch 
from torch import nn 
from torch.nn import functional as F 

# 定义 VAE 模型
class SentimentVAE(nn.Module): 
    def __init__(self, input_dim, hidden_dim, latent_dim, sentiment_dim): 
        super(SentimentVAE, self).__init__() 
        
        # 对于编码器，可以使用 RNN、LSTM 和 Transformer 等时序模型进行设计
        self.encoder = nn.LSTM(input_dim, hidden_dim) 
        
        # 将编码器的输出转换为潜在空间的均值和方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim) 
        self.fc_var = nn.Linear(hidden_dim, latent_dim) 

        # 解码器
        self.decoder = nn.LSTM(latent_dim + sentiment_dim, hidden_dim) 
        
        # 最后的全连接层
        self.fc_output = nn.Linear(hidden_dim, input_dim) 

    def reparameterize(self, mu, log_var): 
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std) 
        return mu + eps*std 

    def forward(self, x, sentiment): 
        # 编码器
        hidden, _ = self.encoder(x) 
        # 得到潜在空间的均值和方差
        mu, log_var = self.fc_mu(hidden), self.fc_var(hidden) 
        # 重参数化技巧
        z = self.reparameterize(mu, log_var) 
        # 将潜在表示和情感信息拼接
        z = torch.cat((z, sentiment), dim=1) 
        # 解码器
        out, _ = self.decoder(z) 
        out = self.fc_output(out) 
        return out, mu, log_var