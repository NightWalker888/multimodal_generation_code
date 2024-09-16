from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline 
import torch 
# 下载并加载 SDXL 1.0 的 Base 模型，若未来 SDXL 模型更新版本，需要根据实际情况替换版本号
pipe = StableDiffusionXLPipeline.from_pretrained( 
 "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, 
 variant="fp16", use_safetensors=True 
) 
pipe.to("cuda") 
# 下载并加载 SDXL 1.0 的 Refiner 模型，若未来 SDXL 模型更新版本，需要根据实际情况替换版本号
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained( 
 "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, 
 use_safetensors=True, variant="fp16" 
) 
refiner.to("cuda") 
prompt = "ultra close-up color photo portrait of a lovely corgi" 
use_refiner = True 
# 使用 Base 模型生成图像
image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil"). 
images[0] 
# 使用 Refiner 模型生成图像
image = refiner(prompt=prompt, image=image[None, :]).images[0]