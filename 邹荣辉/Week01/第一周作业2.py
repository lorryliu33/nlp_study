import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from openai import OpenAI

dataset=pd.read_csv('C:\\Users\\Administrator\\Desktop\\研究生\\ai2026\\AI大模型学习\\Week01\\dataset.csv',sep="\t",header=None,nrows=None)

data=dataset[0].apply(lambda x:" ".join(jieba.lcut(x)))

vector=CountVectorizer()
vector.fit(data.values)
data1=vector.transform(data.values)

model=KNeighborsClassifier()
model.fit(data1,dataset[1].values)

model1=DecisionTreeClassifier()
model1.fit(data1,dataset[1].values)

model2=svm.LinearSVC()
model2.fit(data1,dataset[1].values)


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-c4bedb7585d0444cad8945826efb2aeb", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_classify_ml_knn(text:str)->str:
    """
    Knn进行文本分类，输入文本完成类别划分
    :param text:
    :return:
    """
    text_sentence=" ".join(jieba.lcut(text))
    text_feature=vector.transform([text_sentence])
    return model.predict(text_feature)[0]

def text_classify_ml_tree(text:str)->str:
    """
    决策树进行文本分类，输入文本完成类别划分
    :param text:
    :return:
    """
    text_sentence=" ".join(jieba.lcut(text))
    text_feature=vector.transform([text_sentence])
    return model1.predict(text_feature)[0]


def text_classify_ml_svm(text:str)->str:
    """
    支持向量机SVM进行文本分类，输入文本完成类别划分
    :param text:
    :return:
    """
    text_sentence=" ".join(jieba.lcut(text))
    text_feature=vector.transform([text_sentence])
    return model2.predict(text_feature)[0]

def text_classify_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型的代号

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

if __name__ == "__main__":
    print("Knn分类结果：",text_classify_ml_knn("放一首周杰伦的稻香"))
    print("决策树分类结果：",text_classify_ml_tree("今天天气怎么样？"))
    print("支持向量机分类结果：",text_classify_ml_svm("设置早上9点的闹钟！"))
    print("千问大模型分类结果：",text_classify_llm("播放电影《阿凡达2》"))
