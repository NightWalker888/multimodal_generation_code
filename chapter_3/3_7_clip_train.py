# image_encoder：图像编码器，可以使用 ResNet 或者 Vision Transformer 结构
# text_encoder:文本编码器，可以使用CBOW（Continuous Bag of Words）或者Text Transformer结构
# I[n, h, w, c]：一个训练批次的图像
# T[n, l]：一个训练批次的对应文本描述
# W_i[d_i, d_e]：可学习的图像特征映射
# W_t[d_t, d_e]：可学习的文本特征映射
# t：一个可学习的温度系数
 
# 步骤 1：提取图像模态和文本模态的特征
I_f = image_encoder(I) #[n, d_i] 
T_f = text_encoder(T) #[n, d_t] 
# 步骤 2：将图像特征和文本特征分别映射到共同的多模态空间 [n, d_e] 
# 同时，对这两个多模态特征向量进行归一化
I_e = l2_normalize(np.dot(I_f, W_i), axis=1) 
T_e = l2_normalize(np.dot(T_f, W_t), axis=1) 
# 步骤 3：计算余弦距离 [n, n] 
logits = np.dot(I_e, T_e.T) * np.exp(t) 
# 步骤 4：计算损失
labels = np.arange(n) 
loss_i = cross_entropy_loss(logits, labels, axis=0) 
loss_t = cross_entropy_loss(logits, labels, axis=1) 
loss = (loss_i + loss_t)/2