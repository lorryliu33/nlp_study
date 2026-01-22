import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset.head(5))

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) 

vector = CountVectorizer() 
vector.fit(input_sententce.values) 
input_feature = vector.transform(input_sententce.values) # 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
print(model)

test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))
