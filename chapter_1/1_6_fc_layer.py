import numpy as np

# 假设N是输入特征数量，M是输出特征数量
N = 5  # 输入特征数量
M = 3  # 输出特征数量

# 随机生成输入数据x、权重W和偏置b
x = np.random.rand(N)     # 输入向量(N,)
W = np.random.rand(M, N)  # 权重矩阵(M, N)
b = np.random.rand(M)     # 偏置向量(M,)

# 计算Wx+b
output = np.dot(W, x) + b  # 输出向量(M,)

print(output)