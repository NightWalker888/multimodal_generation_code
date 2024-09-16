from PIL import Image 
from io import BytesIO 
from datasets import load_dataset 
import os 
from tqdm import tqdm 

dataset = load_dataset("nelorth/oxford-flowers") 
# 创建一个用于保存图像的文件夹
images_dir = "./oxford-datasets/raw-images" 
os.makedirs(images_dir, exist_ok=True) 

# 针对 oxford-flowers，遍历并保存所有图像，整个过程持续 15 min 左右
for split in dataset.keys(): 
    for index, item in enumerate(tqdm(dataset[split])): 
        image = item['image'] 
        image.save(os.path.join(images_dir, f"{split}_image_{index}.png"))