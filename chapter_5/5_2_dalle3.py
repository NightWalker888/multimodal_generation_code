from openai import OpenAI 
client = OpenAI() 
# 使用“文生图”功能
response = client.images.generate(
 model="dall-e-3", 
 prompt="A dragon fruit wearing karate belt in the snow", 
 size="1024x1024", 
 quality="standard", #如使用 hd 模式，将消耗更多词符
 n=1, 
) 
image_url = response.data[0].url