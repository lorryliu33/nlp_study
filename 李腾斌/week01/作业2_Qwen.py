"""
使用dataset.csv数据集完成文本分类操作，需要尝试2种不同的模型
"""
from openai import OpenAI

# 使用Qwen大模型来实现

client = OpenAI(
    api_key="sk-f*****f851dda",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def using_llm(text: str):
    try:
        completion = client.chat.completions.create(
            model="qwen-flash",
            messages=[
                {"role": "user",
                 "content": f"帮我进行文本分类: {text}"
                            f"输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。"
                            f"FilmTele-Play,Video-Play,Music-Play,Radio-Listen,Alarm-Update,Travel-Query,HomeAppliance-Control,Weather-Query,Calendar-Query,TVProgram-Play,Audio-Play,Other"},
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("Qwen调用失败:", e)
        return ""

if __name__ == "__main__":
    print("大语言模型: ", using_llm("帮我导航到天安门"))
