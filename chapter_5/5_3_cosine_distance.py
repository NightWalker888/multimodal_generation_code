import numpy as np 
def cosine_distance(a, b): 
 dot_product = np.dot(a, b) 
 norm_a = np.linalg.norm(a) 
 norm_b = np.linalg.norm(b) 
 cosine_similarity = dot_product / (norm_a * norm_b) 
 
 # 由于精度问题，有时候 cosine_similarity 可能略大于 1，所以使用 clip 进行截取操作
 cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0) 
 
 cosine_distance = 1.0 - cosine_similarity 
 
 return cosine_distance 
# 使用示例
a = np.array([1,2,3]) 
b = np.array([4,5,6]) 
print(cosine_distance(a, b))