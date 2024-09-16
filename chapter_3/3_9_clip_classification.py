import torch 
from PIL import Image 
import open_clip 
import urllib.request 
import matplotlib.pyplot as plt 
# 加载 OpenCLIP 预训练模型
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', 
pretrained='laion2b_s34b_b79k') 
tokenizer = open_clip.get_tokenizer('ViT-B-32') 
# 加载图像并对图像进行预处理
image_url = " http://www.***.com/test" 
image_path = "test_image.png" 
urllib.request.urlretrieve(image_url, image_path) 
image = Image.open(image_path).convert("RGB") 
image = preprocess(image).unsqueeze(0) 
# 定义目标类别
text = tokenizer(["a diagram", "a dog", "a cat"]) 
with torch.no_grad(), torch.cuda.amp.autocast(): 
    # 使用 CLIP 图像编码器对图像进行编码
    image_features = model.encode_image(image) 
    # 使用 CLIP 文本编码器对文本描述进行编码
    text_features = model.encode_text(text) 
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True) 
    # 计算图像特征向量和文本特征向量的相似度分数
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1) 
plt.imshow(Image.open(image_path)) 
plt.show() 
# 打印预测的类别
print(f"prob: a diagram {text_probs[0][0]}, a dog {text_probs[0][1]}, a cat 
{text_probs[0][2]}")