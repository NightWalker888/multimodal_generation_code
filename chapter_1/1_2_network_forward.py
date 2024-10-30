import numpy as np 

# ReLU激活函数
def relu(x):
  return np.maximum(0, x)
  
# 初始化网络参数
def initialize_parameters(input_size, hidden_size1, hidden_size2, output_size):
    parameters = {
        'W1': np.random.randn(hidden_size1, input_size) * 0.01,
        'b1': np.zeros((hidden_size1, 1)),
        'W2': np.random.randn(hidden_size2, hidden_size1) * 0.01,
        'b2': np.zeros((hidden_size2, 1)),
        'W3': np.random.randn(output_size, hidden_size2) * 0.01,
        'b3': np.zeros((output_size, 1))
    }
    return parameters
    
# 网络的前向传播
def forward_pass(X, parameters):
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    A1 = relu(Z1)
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = relu(Z2)
    Z3 = np.dot(parameters['W3'], A2) + parameters['b3']
    A3 = Z3  # 若为分类问题，此处可能改为softmax等激活函数
    cache = (Z1, A1, Z2, A2, Z3, A3)
    return A3, cache
    
# 设置网络参数
input_size = 2  # 输入特征数量
hidden_size1 = 4  # 第一个隐藏层神经元数量
hidden_size2 = 3  # 第二个隐藏层神经元数量
output_size = 1  # 输出层神经元数量

# 初始化网络参数
parameters = initialize_parameters(input_size, hidden_size1, hidden_size2, output_size)

# 假设输入数据
X = np.random.randn(input_size, 1)  # 一个样本的特征

# 前向传播
output, _ = forward_pass(X, parameters)
print("网络输出:", output)
