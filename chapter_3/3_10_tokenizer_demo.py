from transformers import AutoTokenizer, AutoModel
import torch 

# 以bert-base-uncased为例
model_name = "bert-base-uncased"

# 加载预训练模型及其分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 示例文本
text = "Hello world"

# 使用分词器处理文本，获得词符标识
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
print(inputs["input_ids"])

# 获取词嵌入向量
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print(embeddings.shape)
