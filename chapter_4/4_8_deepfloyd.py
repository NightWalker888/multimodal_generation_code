from diffusers import DiffusionPipeline 
from diffusers.utils import pt_to_pil 
import torch 
# 加载 DeepFloyd 第一阶段模型
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant= 
"fp16", torch_dtype=torch.float16) 
#如果 torch.__version__ >= 2.0.0，删除下面这一行
stage_1.enable_xformers_memory_efficient_attention() 
stage_1.enable_model_cpu_offload() 
# 加载 DeepFloyd 第二阶段模型
stage_2 = DiffusionPipeline.from_pretrained( 
 "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", 
torch_dtype=torch.float16 
) 
#如果 torch.__version__ >= 2.0.0，删除下面这一行
stage_2.enable_xformers_memory_efficient_attention() 
stage_2.enable_model_cpu_offload()

# 加载 DeepFloyd 第三阶段模型
safety_modules = {"feature_extractor": stage_1.feature_extractor, 
 "safety_checker": stage_1.safety_checker, "watermarker": 
 stage_1.watermarker} 
stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusionx4-upscaler", **safety_modules, torch_dtype=torch.float16) 
#如果 torch.__version__ >= 2.0.0，删除下面这一行
stage_3.enable_xformers_memory_efficient_attention() 
stage_3.enable_model_cpu_offload()