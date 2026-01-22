import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

df = pd.read_csv('dataset.csv', sep='\t', header=None, nrows=10000)

input_sententce = df[0].apply(lambda x: "".join(jieba.lcut(x)))

vector = CountVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

model = KNeighborsClassifier()
model.fit(input_feature, df[1].values)


client = OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-d3eb3f2cfd354147a78249376ce05718", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 大语言模型
def text_calssify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
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
    return completion.choices[0].message.content

# 机器学习
def text_calssify_using_ml(text: str) -> str:
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = vector.transform([text_sentence])

    return model.predict(text_feature)[0]


if __name__ == '__main__':
    input_text = "播放'我们的歌'"
    # input_text = "导航回家"
    text_llm = text_calssify_using_llm(input_text)
    print("大语言模型: ", text_llm)
    text_ml = text_calssify_using_ml(input_text)
    print("机器学习: ", text_ml)
