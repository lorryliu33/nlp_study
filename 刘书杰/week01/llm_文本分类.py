from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-5392e86d73f74f9b81a485826b1b07f8", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-flash", # 模型的代号

    # 对话列表
    messages=[
        {"role": "user", "content": f"""帮我进行文本分类：帮我播放一下剑来动漫

    输出的类别只能从如下中进行选择， 只输出下面的类别，没有其他赘述的话，请给出最合适的类别。
    FilmTele-Play            
    Video-Play               
    Music-Play              
    Radio-Listen           
    Alarm-Update        
    Travel-Query        
    HomeAppliance-Control  
    Weather-Query          
    Calendar-Query      
    TVProgram-Play      
    Audio-Play       
    Other             
    """},  # 用户的提问
    ]
)
print(completion.choices[0].message.content)
