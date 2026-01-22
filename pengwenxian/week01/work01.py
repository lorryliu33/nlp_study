"""
KNN模型预测
"""
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

## 读取数据集
df = pd.read_csv('dataset.csv', sep='\t', nrows=1000, header=None)
## 对每行的待分类文本进行分词 以空格拼接
text_series = df[0].apply(lambda x: " ".join(jieba.lcut(x)))

## 统计词频
vector = CountVectorizer()
vector.fit(text_series.values)
## 将词进行向量化，将每个文本按照词频进行向量化统计
input_datas = vector.transform(text_series.values)

#创建KNN模型
model = KNeighborsClassifier()
#将分词结果与 文本分类的结果进行对应训练
model.fit(input_datas, df[1].values)

# 创建待预测的文本
classified_text = "帮我播放一下黄昏的歌曲"
# 先分词
classified_seq = " ".join(jieba.lcut(classified_text))
## 将词进行向量化，按照词频进行向量化统计
classified_feature = vector.transform([classified_seq])
print("待预测文本:", classified_text)
print("KNN模型预测结果:",model.predict(classified_feature))