import os

from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key=os.environ["DASHSCOPE_API_KEY"], # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

        messages=[
            {
                "role": "system",
                "content": """你是一个专业的文本分类助手。请根据输入的文本内容将其分类到以下预定义类别之一：
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

                输出格式：只需要返回类别名称，不要有其他额外说明。"""
            },
            {
                "role": "user",
                "content": f"帮我进行文本分类：{text}"
            },  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":

    print("大语言模型: ", text_calssify_using_llm("帮我导航到天安门"))