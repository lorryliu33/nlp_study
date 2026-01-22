import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN

dataset = pd.read_csv("D:\python_code\extension\Week01\dataset.csv", sep="\t", header=None, nrows=100)
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
# print(input_sententce) # 0            还有 双鸭山 到 淮阴 的 汽车票 吗 13 号 的

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 100 * 词表大小 (0, 0) 坐标	1 值
# print(vector.vocabulary_) # 词表 到、的、吗、号 等可能因频率阈值被排除 正则表达式会将他们过滤（长度问题）
# print(input_feature)

model = KNeighborsClassifier() #默认K=5
model.fit(input_feature, dataset[1].values)


test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("待预测的文本的向量", test_feature)
print("KNN模型预测结果: ", model.predict(test_feature))