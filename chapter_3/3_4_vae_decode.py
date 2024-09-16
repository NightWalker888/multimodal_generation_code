from PIL import Image 
import numpy as np 
import torch 
from diffusers import AutoencoderKL 
device = 'cuda' 
# 加载 VAE 模型
vae = AutoencoderKL.from_pretrained( 
 'CompVis/stable-diffusion-v1-4', subfolder='vae') 
vae = vae.to(device) 
 
pths = ["test_imgs/new.png", "test_imgs/full.png"] 
for pth in pths: 
    img = Image.open(pth).convert('RGB') 
    img = img.resize((512, 512)) 
    img_latents = encode_img_latents(img) # 编码，img_latents 的维度是[1,4,64,64] 
    dec_img = decode_img_latents(img_latents)[0] #解码