import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import neighbors
from openai import OpenAI

data = pd.read_csv('dataset.csv', sep='\t', header=None, nrows=1000)
input_sentence = data[0].apply(lambda x: " ".join(jieba.lcut(x)))
vector = CountVectorizer()
vector.fit(input_sentence.values)
output_feature = vector.transform(input_sentence.values)
# print(output_feature.shape)
model = neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(output_feature, data[1].values)

client = OpenAI(
    api_key='sk-9bf45d961ac64f75a3b6a64c7fd08817',

    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
)

def text_classify_using_ml(text):
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = vector.transform([text_sentence])
    # print("模型预测结果:", model.predict(text_feature))
    print("模型预测结果:", model.predict(text_feature)[0])

def text_classify_using_llm(text):
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-max",
        messages=[
            {"role": "user", "content": f"""请帮我给文本分类：{text}
输出的类别只能从如下中进行选择,只输出类型
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
Other """}
        ]
    )
    print("模型预测结果:", completion.choices[0].message.content)

if __name__ == '__main__':
    text_query = "帮我播放一下郭德纲的小品"
    text_classify_using_ml(text_query)
    text_classify_using_llm(text_query)
