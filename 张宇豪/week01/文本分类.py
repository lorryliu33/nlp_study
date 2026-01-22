import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI
from typing import Union
from fastapi import FastAPI


app = FastAPI()

dataset = pd.read_csv('dataset.csv', sep='\t', header=None, nrows=None)
input_sentense = dataset[0].apply(lambda x : ' '.join(jieba.lcut(x)))

vector = CountVectorizer()
input_feature = vector.fit_transform(input_sentense.values) # 对划分后的文本词语进行编号并转换成向量

model = KNeighborsClassifier(n_neighbors=3) # KNN3
model.fit(input_feature, dataset[1].values)

client = OpenAI(
    api_key='sk-08a6acbb4a2e46e195b8199036824588',

    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
)

@app.get("/text-cls/ml")
def text_classify_using_ml(text: str) -> str:
    """
    用机器学习进行文本分类
    """
    text_feature = vector.transform([' '.join(jieba.lcut(text))])
    return model.predict(text_feature)[0]

@app.get("/text-cls/llm")
def text_classify_using_llm(text: str) -> str:
    """
    用大模型进行文本分类
    """
    completion = client.chat.completions.create(
        model='qwen-flash',

        messages=[
            {'role': 'system',
             'content': '你是文本分类助手，需要对我输入的文本进行分类, 只输出类别即可，其他的东西一概不要输出'},
            {'role': 'user', 'content': f'帮我判断"{text}"这句话是什么类别，类别只能从：'
                                        'FilmTele-Play, Video-Play, Music-Play, Radio-Listen, Alarm-Update,'
                                        ' Weather-Query, Travel-Query, HomeAppliance-Control, Calendar-Query, TVProgram-Play, Audio-Play, Other这几个类别中选择'}
        ]
    )
    return completion.choices[0].message.content
#
# if __name__ == '__main__':
#     print("机器学习：", text_classify_using_ml("我想听王菲的《匆匆那年》"))
#     print("大模型：", text_classify_using_llm("我想听王菲的《匆匆那年》"))
