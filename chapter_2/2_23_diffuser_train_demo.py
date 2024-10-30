import torch 
from datasets import load_dataset 
from torchvision import transforms 
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline 
import torch.nn.functional as F 
import accelerator 
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np 


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MapDataset(Dataset):
    def __init__(self, dataset, map_func):
        """
        初始化 MapDataset
        :param dataset: 原始数据集
        :param map_func: 映射函数，应用于数据集中的每个样本
        """
        self.dataset = dataset
        self.map_func = map_func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 获取原始数据集中的样本
        item = self.dataset[index]
        # 应用映射函数
        return self.map_func(item)


# 定义预处理步骤
preprocess = transforms.Compose( 
    [ 
        transforms.Resize((128, 128)), 
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5]), 
    ] 
)

# 应用预处理转换
def preprocess_data(example):
    image = example['image']
    image = preprocess(image)
    example['image'] = image
    return example


# 模型推理
def inference_test(model, epoch, step):
    pipeline = DDPMPipeline(unet = model, scheduler = noise_scheduler)

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    images = pipeline(
        generator=generator,
        batch_size=4,
        num_inference_steps=50,
        output_type="np"
    ).images

    plt.imshow(np.concatenate(images,1))  # 显示第一张图像
    plt.savefig(f'generated_image_epoch_{epoch}_step_{step}.png')
    del pipeline 

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on device: {device}")

    # 加载数据集
    dataset = load_dataset("nelorth/oxford-flowers", split="train") 
    new_dataset = MyDataset(dataset)

    # 使用map方法应用预处理函数
    dataset_mapped = MapDataset(new_dataset, preprocess_data)

    # 将数据集封装成训练用的格式
    train_dataloader = torch.utils.data.DataLoader(dataset_mapped, batch_size=16, shuffle=True)

    # 定义模型结构 

    model = UNet2DModel( 
        sample_size=128, # 目标图像的分辨率
        in_channels=3, # 输入通道的数量，对于 RGB 图像，此值为 3 
        out_channels=3, # 输出通道的数量
        layers_per_block=2, # 每个 U-Net 模型中使用的 ResNet 层的数量
        block_out_channels=(128, 128, 256, 256, 512, 512), # 每个 U-Net 模型的输出通道的数量
        down_block_types=( 
        "DownBlock2D", # 常规的 ResNet 下采样模块
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D", # 具有空间自注意力机制的 ResNet 下采样模块
        "DownBlock2D", 
        ), 
        up_block_types=( 
        "UpBlock2D", # 常规的 ResNet 上采样模块
        "AttnUpBlock2D", # 具有空间自注意力机制的 ResNet 上采样模块
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D" 
        ), 
    )

    model.to(device)

    num_epochs = 10

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr= 1e-4,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )

    for epoch in range(num_epochs): 
        for step, batch in enumerate(train_dataloader): 
            clean_images = batch['image'].to(device)
            # 对应扩散模型训练过程：随机采样噪声
            noise = torch.randn(clean_images.shape).to(clean_images.device) 
            bs = clean_images.shape[0] 
            # 对应扩散模型训练过程：对 batch 中的每张图像，随机选取时间步 t 
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), 
            device=clean_images.device).long() 
            # 对应扩散模型训练过程：一次计算加噪结果
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps) 

            # 对应扩散模型训练过程：预测噪声并计算损失函数
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0] 
            loss = F.mse_loss(noise_pred, noise)

            if step % 10 == 0: 
                print(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item()}")

            if step % 100 == 0:
                inference_test(model, epoch, step) 

            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward() 
            optimizer.step()


    # 保存模型
    torch.save(model.state_dict(), "diffusion_train_model_new.pth")
