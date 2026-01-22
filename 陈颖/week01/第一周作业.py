import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.tree import DecisionTreeClassifier #DecisionTree


dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型（不定长文本转化为维度相同向量，统计词频）
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 100 * 词表大小

knn_model = KNeighborsClassifier()
knn_model.fit(input_feature, dataset[1].values)

dt_model = DecisionTreeClassifier()
dt_model.fit(input_feature, dataset[1].values)

def text_calssify_using_knn(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return knn_model.predict(test_feature)[0]

def text_calssify_using_dt(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return dt_model.predict(test_feature)[0]

def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    pass

if __name__ == "__main__":

    #pandas 用来进行表格的加载和分析
    #numpy 从矩阵的角度进行加载和计算
    print("KNN: ", text_calssify_using_knn("帮我导航到天安门"))
    print("DT: ", text_calssify_using_dt("帮我导航到天安门"))
