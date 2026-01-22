import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.naive_bayes import MultinomialNB # 朴素贝叶斯分类器

# 第一步： 读取数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, names=["Text","Label"], nrows=10000)
print(dataset.head(5))

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset["Text"].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 100 * 词表大小
print(f"输入特征的形状: {input_feature.shape}")

# 第二步： 构建模型并训练
# model = KNeighborsClassifier() # KNN模型初始化

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=500) # 仅加一个参数，解决收敛警告

from sklearn.svm import LinearSVC
model = LinearSVC() 

# model = MultinomialNB() # 朴素贝叶斯模型初始化
model.fit(input_feature, dataset["Label"].values) # 训练模型
print(model)

# 第三步： 预测
test_query = "帮我查询一下今天的天气"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("模型预测结果: ", model.predict(test_feature))
