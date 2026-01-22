"""
调用千问模型进行预测
"""
import pandas as pd
from openai import OpenAI
## 获取数据源
df = pd.read_csv('dataset.csv', sep="\t", header=None)

## 获取所有的分类，用于发送给AI来分类
all_type_list_str = "[" + "、".join(list(set(df[1].values.tolist()))) + "]"

## 根据token创建AI的连接
client = OpenAI(
    api_key="sk-89beb5cc538544fc9ab0ad56bcf6f044",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

## 创建待分类文本
text = "帮我播放一下黄昏的歌曲"

## 创建基于qwen-flash的模型，输入命令
res = client.chat.completions.create(
    model="qwen-flash",
    messages=[{"role": "user",
               "content": f"""请帮我分类{text}这个文本，所有的分类种类如下:{all_type_list_str}。且你只需要输出类型给我即可。"""}]
    )
print(res.choices[0].message.content)