import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

"""一、机器学习实现文本分类"""
# 1、加载数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(len(dataset))
print(dataset.shape)
print(dataset.head(5))
print(dataset.tail(5))
print(set(dataset[1]))


# 2、特征提取
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
print(input_sentence[0])
vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)
print(type(input_feature))


# 3、模型训练
# model = RandomForestClassifier()
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
print(model)


# 4、进行预测
test = "帮我播放一首刘德华的歌"
test_sentence = " ".join(jieba.lcut(test))
print(test_sentence)
test_feature = vector.transform([test_sentence])
print(f'带预测文本：{test}')
print(f'机器学习-预测结果：{model.predict(test_feature)[0]}')


"""二、大模型实现文本分类"""
client = OpenAI(
    # api_key
    api_key='sk-f0172bf78090473ba143e712a4b18ee9',
    # 大模型厂商地址，阿里云
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

completion = client.chat.completions.create(
    model='qwen-flash',

    messages=[
        {"role": "user", "content": f"对 {test} 进行文本分类，所分的类别只能在{set(dataset[1])}中，回答文案只需要对应的类别"}
    ]
)
print(f"大模型-预测结果：{completion.choices[0].message.content}")
