from zai import ZhipuAiClient

system_config = {
    "system_prompt": """给你一段文本，请帮我进行文本分类，输出的类别只能从如下中进行选择：

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

除了类别之外下列的类别，请给出最合适的类别。""",
    "examples": [
        {"role": "user", "content": "我想看一个美国探险纪录片。"},
        {"role": "assistant", "content": "Video-Play"},
    ],
    "temperature": 0.3
}

# API_KEY = "04503c3c1ea845d9934516c87c684d12.Gu2MCAbs2Ze2WAXG"  # 智谱api_key
client = ZhipuAiClient(api_key= "API_KEY") 

def call_model(input_text: str): 
    try:
        response = client.chat.completions.create(
            model="glm-4.7",
            messages=[
                {"role": "system", "content": system_config["system_prompt"]},
                *system_config["examples"],
                {"role": "user", "content": input_text},
            ],
            thinking={
                "type": "disabled",    # 关闭深度思考模式
            },
            max_tokens=128,          # 最大输出 tokens
            temperature=system_config["temperature"]           # 控制输出的随机性
        )
        result = response.choices[0].message.content.strip()
        print(f"模型分类结果: {result}")
        return result
    except Exception as e:
        print(f"调用模型失败，错误信息：{str(e)}")
        return None

while True:
    user_input = input("请输入要分类的文本（输入 'exit' 退出）：")

    if not user_input:
            print("输入内容不能为空，请重新输入！\n")
            continue

    if user_input.lower() in ['exit','quit','q']:
        print("\n程序已退出！")
        break
    
    call_model(user_input)

