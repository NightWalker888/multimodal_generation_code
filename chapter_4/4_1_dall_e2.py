from openai import OpenAI 
client = OpenAI() 
# 使用“文生图”功能
response = client.images.generate( 
    model="dall-e-2", 
    prompt="A dragon fruit wearing karate belt in the snow", 
    prompt="A robot couple fine dining with Eiffel Tower in the background" 
    size="512x512", 
    quality="standard", #"hd", # "standard" 
    n=1, 
) 
image_url = response.data[0].url