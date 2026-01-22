import jieba
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer #词频统计
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=1000)
# print(dataset.head(5))

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values) #统计词表
input_feature = vector.transform(input_sententce.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
##
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-ktdvmzqutapsrdedjnvnsctkqivkapsgrjjmpgavyzxgfdos", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    base_url="https://api.siliconflow.cn/v1"
)

def ml_model(test_query):
    # test_query = "帮我播放一下郭德纲的小品"
    test_sentence = " ".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])
    print("待预测的文本", test_query)
    print("ML模型预测结果: ", model.predict(test_feature))

def llm_model(test_query):
    completion = client.chat.completions.create(
        # model="qwen-flash",  # 模型的代号
        model="deepseek-ai/DeepSeek-V3.1-Terminus",

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{test_query}

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
    # return completion.choices[0].message.content
    # print("待预测的文本", test_query)
    print("LLM模型预测结果: ", completion.choices[0].message.content)



if __name__ == '__main__':
    ml_model("明天天气怎么样")
    llm_model("明天天气怎么样")
    # data = pd.read_csv("dataset.csv",sep = '\t',names = ['text','label'],nrows=100)
    # print(data.head(10))
    # print(data.shape)
    #
    # print(data[])
