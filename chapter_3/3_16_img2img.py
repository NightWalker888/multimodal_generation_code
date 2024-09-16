import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline 

# 用于将多张图像拼接
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# 加载一个Stable Diffusion模型，这里使用名为ToonYou的模型进行图像风格化
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("zhyemmmm/ToonYou")
pipe = pipe.to(device)

# 下载蒙娜丽莎的图片
url = "https://ice.frostsky.com/2023/08/26/2c809fbfcb030dd8a97af3759f37ee29.png"#
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((512, 512))

prompt = "1girl, fashion photography"
images = []

# 设置不同的重绘强度参数，比较图生图效果
for strength in [0.05, 0.15, 0.25, 0.35, 0.5, 0.75]:
  image = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=7.5).images[0]
  images.append(image)

# 可视化图像
result_image = image_grid(images, 2, 3)
result_image.save("img2img.jpg")