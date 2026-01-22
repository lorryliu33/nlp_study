"""
使用dataset.csv数据集完成文本分类操作，需要尝试2种不同的模型
"""
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 使用机器学习（Scikit-learn）中的KNN分类器来实现

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=None)
print(dataset.head(10))

def segment_text(text: str):
    return " ".join(jieba.lcut(text)) # jieba分词，并用空格连接
input_sentence = dataset[0].apply(segment_text) # 对dataset[0]的每条文本都进行segment_text处理

vector = CountVectorizer() # 把文本转换成词频向量
vector.fit(input_sentence.values) # 构建词典，学习文本中出现过的所有词
print(vector.vocabulary_)

# 将每条文本转换成稀疏向量，向量长度 = 词典大小
# 每个位置存的是该词在文本中出现的次数
# input_feature形状 (样本数, 词典大小)
input_feature = vector.transform(input_sentence.values)
print(input_feature)

model = KNeighborsClassifier() # KNN
# dataset[1].values：文本对应的标签，也就是目标值列
# fit() KNN模型存储训练样本和标签，准备预测
model.fit(input_feature, dataset[1].values)
print(model)

test_query = "查找刘谦的春晚节目视频"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))






# 使用Qwen大模型来实现

# 使用？？？大模型来实现
