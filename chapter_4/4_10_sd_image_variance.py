from diffusers import StableUnCLIPImg2ImgPipeline 
from diffusers.utils import load_image 
import torch 
# 加载 Stable Diffusion 图像变体模型
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained( 
 "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, 
 variation="fp16") 
pipe = pipe.to("cuda")

# 可以使用其他需要测试的图像的 URL 链接
url = "http://***.com/test.png" 
init_image = load_image(url) 
# 生成图像变体
images = pipe(init_image).images 
images[0].save("variation_image.png")