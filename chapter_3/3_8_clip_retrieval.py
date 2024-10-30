import torch 
import clip 
from PIL import Image 
import urllib.request 
import matplotlib.pyplot as plt 

# 加载 CLIP 预训练模型
device = "cuda" if torch.cuda.is_available() else "cpu" 
model, preprocess = clip.load("ViT-B/32", device=device) 

# 定义目标类别
target_classes = ["cat", "dog"] 

# 加载图像并对图像进行预处理
image_url = " http://www.***.com/test" 
image_path = "test_image.png" 
urllib.request.urlretrieve(image_url, image_path) 
image = Image.open(image_path).convert("RGB") 
image_input = preprocess(image).unsqueeze(0).to(device) 

# 使用 CLIP 图像编码器对图像进行编码
with torch.no_grad(): 
    image_features = model.encode_image(image_input) 
 
# 使用 CLIP 文本编码器对文本描述进行编码
text_inputs = clip.tokenize(target_classes).to(device) 

with torch.no_grad(): 
    text_features = model.encode_text(text_inputs) 
 
# 计算图像特征向量和文本特征向量的相似度分数
similarity_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1) 
# 获取相似度分数最大的文本特征向量，确定分类类别
_, predicted_class = similarity_scores.max(dim=-1) 
predicted_class = predicted_class.item() 
# 打印预测的类别
predicted_label = target_classes[predicted_class] 
plt.imshow(image) 
plt.show() 
print(f"Predicted class: {predicted_label}") 
print(f"prob: cat {similarity_scores[0][0]}, dog {similarity_scores[0][1]}")