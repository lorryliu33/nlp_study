import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF
from sklearn.neighbors import KNeighborsClassifier # KNN
from fastapi import  FastAPI
from sklearn import  tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from openai import  OpenAI

app=FastAPI()

dataset = pd.read_csv("./dataset.csv",sep='\t', header=None, nrows=1000) #header是代表有没有列名，nrow是读取行训练行数  sep可用正则

# print(dataset.head())
print("数据集样本维度",dataset.shape)#维度是指矩阵大小说明
print("频次统计，",dataset[1].value_counts())

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理\

#TF
vector = CountVectorizer() # 创建特征提取器 tf
vector.fit(input_sententce, dataset[1].values) #学习
feature = vector.transform(input_sententce.values) #词频统计 转换为矩阵

#TF-IDF
tfVector=TfidfVectorizer()
tfVector.fit(input_sententce, dataset[1].values)
tfFeature = tfVector.transform(input_sententce.values)

#knn模型
model=KNeighborsClassifier()
model.fit(tfFeature,dataset[1].values)
# model.fit(feature,dataset[1].values)

#tree模型
treeModel=tree.DecisionTreeClassifier()
treeModel.fit(feature,dataset[1].values)

#随机森林
randomModel=RandomForestClassifier()
randomModel.fit(feature,dataset[1].values)

#朴素贝叶斯
bayesModel= MultinomialNB() #MultinomialNB是朴素贝叶斯的Classifier
bayesModel.fit(feature,dataset[1].values)


client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        # https://bailian.console.aliyun.com/?tab=model#/api-key
        api_key="sk-a15044896da345f6ad8cf3af5fb60f2a",  # 账号绑定，用来计费的
        # 大模型厂商的地址，阿里云
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

# @app.get("/knnml")
def text_calssify_using_knn_ml(text:str):
    sententce=" ".join((jieba.lcut(text)))
    text_feature=  vector.transform([sententce])
    return  model.predict(text_feature)[0]

# @app.get("/treeml")
def text_calssify_using_tree_ml(text:str):
    sententce=" ".join((jieba.lcut(text)))
    text_feature=  vector.transform([sententce])
    return  treeModel.predict(text_feature)[0]

# @app.get("/randomml")
def text_calssify_using_random_ml(text:str):
    sententce=" ".join((jieba.lcut(text)))
    text_feature=  vector.transform([sententce])
    return  randomModel.predict(text_feature)[0]

# @app.get("/bayesml")
def text_calssify_using_bayes_ml(text:str):
    sententce=" ".join((jieba.lcut(text)))
    text_feature=  vector.transform([sententce])
    return  bayesModel.predict(text_feature)[0]

# @app.get("/qwflashllm")
def text_calssify_using_qw_llm(text:str)->str:
    # | `role` | 含义 | 说明 |
    # | -------------- | ---------------- | ------ |
    # | `system` | 系统设定 | 仅用于初始化，定义模型行为 |
    # | `user` | 用户输入 | 用户说的话，模型要回应 |
    # | `assistant` | 模型回复 | 模型生成的回答 |
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号
    # 对话列表
        messages=[ {"role":"user","content":f"""帮我进行文本分类{text}
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
        """}
        ]
    )
    return  completion.choices[0].message.content

if  __name__=="__main__":
    print("knn机器学习简单词频: ", text_calssify_using_knn_ml("我喜欢电脑游戏英雄联盟"))
    print("tree机器学习简单词频: ", text_calssify_using_tree_ml("我喜欢电脑游戏英雄联盟"))
    print("random机器学习简单词频: ", text_calssify_using_random_ml("我喜欢电脑游戏英雄联盟"))
    print("朴素贝叶斯机器学习简单词频: ", text_calssify_using_bayes_ml("我喜欢电脑游戏英雄联盟"))
    print("qs-flash: ", text_calssify_using_qw_llm("我喜欢电脑游戏英雄联盟"))
