from openai import OpenAI 
client = OpenAI() 
# 使用图像变体功能
response = client.images.create_variation( 
    image=open("你的图像路径", "rb"),
    n=2, 
    size="1024x1024" 
)
image_url = response.data[0].url