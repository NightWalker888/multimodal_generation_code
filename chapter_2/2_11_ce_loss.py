import numpy as np 

def cross_entropy_classification(y_true, y_pred): 
    """ 
    y_true：真实标签。这是任务的真实结果，通常由人类标注或事先已知。
    对于图像分类任务（如猫、狗分类），y_true 可以是类别的索引或 one-hot 编码表示。
    y_pred：预测标签。这是模型预测的结果。
    对于图像分类任务，y_pred 是一个概率分布向量，表示每个类别的预测概率。
    """ 
    # 数值稳定性处理，将预测标签限制在[1e-9, 1-1e-9]内
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9) 
    return -np.sum(y_true * np.log(y_pred)) 

def cross_entropy_segmentation(y_true, y_pred): 
    """ 
    y_true：真实标签。这是任务的真实结果，通常由人类标注或事先已知。
    对于图像分割任务（如语义分割），y_true 是一个二维或多维数组，
    是每个像素对应的类别索引或 one-hot 编码表示。
    y_pred：预测标签。这是模型预测的结果。
    对于图像分割任务，y_pred 是一个三维数组，用于存储每个类别在每个像素的预测概率。
    """ 
    # 数值稳定性处理，将预测标签限制在[1e-9, 1-1e-9]内
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9) 
    num_classes, height, width = y_true.shape 
    total_loss = 0 
    for c in range(num_classes): 
        for i in range(height): 
            for j in range(width): 
                total_loss += y_true[c, i, j] * np.log(y_pred[c, i, j]) 
    return -total_loss 

# 示例代码（假设类别是经过 one-hot 编码的）
y_true_class = np.array([0, 1, 0]) 
y_pred_class = np.array([0.1, 0.8, 0.1]) 
y_true_segment = np.random.randint(0, 2, (3, 32, 32)) 
y_pred_segment = np.random.rand(3, 32, 32) 
# 计算图像分类任务损失
classification_loss = cross_entropy_classification(y_true_class, y_pred_class) 
# 计算图像分割任务损失
segmentation_loss = cross_entropy_segmentation(y_true_segment, y_pred_segment) 
print("图像分类任务损失:", classification_loss) 
print("图像分割任务损失:", segmentation_loss)