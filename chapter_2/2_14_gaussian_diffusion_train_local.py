from PIL import Image 
import torchvision.transforms as transforms 
import torch 

# 预设一个变换操作，将 PIL Image 转换为 PyTorch Tensor，并对其进行归一化
transform = transforms.Compose([ 
 transforms.Resize((128, 128)), 
 transforms.ToTensor(), 
]) 

# 假设有一个列表，其中包含 8 张图像的路径
image_paths = ['path_to_your_image1', 'path_to_your_image2', 
 'path_to_your_image3', 'path_to_your_image4', 
 'path_to_your_image5', 'path_to_your_image6', 
 'path_to_your_image7', 'path_to_your_image8'] 

# 使用列表压缩读取并处理这些图像
images = [transform(Image.open(image_path)) for image_path in image_paths] 
'''将处理好的图像列表转换为一个 4D Tensor，注意 torch.stack 能够自动处理 3D Tensor 到
4D Tensor 的转换'''
training_images = torch.stack(images) 
# 现在 training_images 中应该有 8 张 3×128×128 的图像
print(training_images.shape) # torch.Size([8, 3, 128, 128])