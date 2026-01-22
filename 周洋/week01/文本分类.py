# 1. 导入库
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

# 2. 读取数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(f"数据形状: {dataset.shape}")  # 查看数据维度

# 3. 查看标签分布
label_counts = dataset[1].value_counts()
print("标签分布:")
print(label_counts)
print(f"类别数量: {len(label_counts)}")

# 4. 中文分词
print("正在进行中文分词...")
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(str(x))))
print(f"示例分词结果: {input_sentence.iloc[0]}")  # 查看第一个样本的分词结果

# 5. 特征提取
vector = CountVectorizer(max_features=5000)  # 限制最大特征数为5000
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)
print(f"特征矩阵形状: {input_feature.shape}")  # 应该是 (10000, 5000)

# 6. 训练模型
model = KNeighborsClassifier(n_neighbors=5)  # 设置k=5
model.fit(input_feature, dataset[1].values)
print("模型训练完成!")

client = OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-69021040fed2424c91bdca07a03ec78b",
    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
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

if __name__ == "__main__":

    print("机器学习: ", text_calssify_using_ml("武汉今天风大吗"))
    print("大语言模型: ", text_calssify_using_llm("武汉今天风大吗"))
