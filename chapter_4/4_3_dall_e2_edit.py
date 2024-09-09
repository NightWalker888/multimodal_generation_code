from openai import OpenAI 
client = OpenAI() 

response = client.images.edit(
    model="dall-e-2", 
    image=open("lake.png", "rb"), # 图 4-3 左侧图
    mask=open("lake_mask.png", "rb"), # 图 4-3 中间图
    prompt="a lake with a wooden boat", 
    n=1, 
    size="1024x1024" 
) 
image_url = response.data[0].url