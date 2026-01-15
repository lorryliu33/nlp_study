import jieba
from openai import OpenAI

jieba.setLogLevel('ERROR')  # 屏蔽日志输出

client = OpenAI(
    api_key="sk-dd9265f11a92a9a88dae4fa7527d6cdd0ec0cae284e5da183069c448aa40ebfb", # 账号绑定，用来计费的

    # 大模型厂商的地址，七牛
    base_url="https://api.qnaigc.com/v1",
)


def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model = "deepseek-v3.1",  # 模型的代号

        messages = [
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择， 不要输出其他任何内容不相关的内容。
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
"""}
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":

    print("大语言模型: ", text_calssify_using_llm("帮我导航到天安门"))
