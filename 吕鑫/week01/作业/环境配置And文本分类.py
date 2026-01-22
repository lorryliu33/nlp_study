import jieba
import pandas as pd
from fastapi import FastAPI
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()
dataSource = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataSource[1].value_counts())
# 建模
# 使用空格对语句进行分割
sklearnData = dataSource[0].apply(lambda x: " ".join(jieba.lcut(x)))

countVector = CountVectorizer()
countVector.fit(sklearnData.values)
# 10000*X(词语个数)
dataSourceFeature = countVector.transform(sklearnData.values)

# KNN模型训练
model = KNeighborsClassifier()
model.fit(dataSourceFeature, dataSource[1].values)

client = OpenAI(
    api_key="sk-35609c8a5c4e42c6bc8b38888615c54b", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def text_class_ml(content: str) -> str:
    """
    机器学习语言文本分类
    :param content: 分类的参数内容
    :return: 分类结果
    """
    textSplit = " ".join(jieba.lcut(content))
    # print(textSplit)
    textFeature = countVector.transform([textSplit])
    return model.predict(textFeature)[0]


def text_class_llm(content: str) -> str:
    """
    大预言模型文本分类
    :param content: 分类的参数内容
    :return: 分类结果
    """
    completion = client.chat.completions.create(model="qwen-flash", messages=[
        {"role": "user", "content": f"""帮我进行文本分类：{content}

输出的类别只能从如下中进行选择， 除了类别之外不需要额外信息，请给出最合适的类别。
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
"""},
    ])
    return completion.choices[0].message.content


if __name__ == "__main__":
    # data = pd.read_csv("dataset.csv",sep="\t",nrows=50)
    # print(data.head(10))
    #
    # # 维度
    # print(data.shape)
    # print(data["label"].value_counts())
    # print(text_class_ml("导航回到山西"))
    print(text_class_llm("导航回到山西"))
