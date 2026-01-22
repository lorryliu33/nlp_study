import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("../dataset.csv", sep="\t", header=None, nrows=8000)
print(dataset[1].value_counts())
print(dataset.size)
print(dataset[0].size)

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = CountVectorizer()
input_sentence_values = input_sentence.values
vector.fit(input_sentence_values)
intput_feature_vector = vector.transform(input_sentence_values)
intput_feature_vector2 = vector.fit_transform(input_sentence_values)

model = KNeighborsClassifier()
dataset_ = dataset[1]
values = dataset_.values
model.fit(intput_feature_vector, values)

test_str_1 = "导航带我去光谷步行街"
test_str_1_sentence = " ".join(jieba.lcut(test_str_1))
test_feature_1 = vector.transform([test_str_1_sentence])
print("文本", test_str_1)
print("分类结果: ", model.predict(test_feature_1))

test_str_2 = "带我去光谷步行街"
test_str_2_sentence = " ".join(jieba.lcut(test_str_2))
test_feature_2 = vector.transform([test_str_2_sentence])
print("文本", test_str_2)
print("分类结果: ", model.predict(test_feature_2))


test_str_3 = "给我三万块钱"
test_str_3_sentence = " ".join(jieba.lcut(test_str_3))
test_feature_3 = vector.transform([test_str_3_sentence])
print("文本", test_str_3)
print("分类结果: ", model.predict(test_feature_3))

test_str_3 = "快点儿打开厨房的油烟机"
test_str_3_sentence = " ".join(jieba.lcut(test_str_3))
test_feature_3 = vector.transform([test_str_3_sentence])
print("文本", test_str_3)
print("分类结果: ", model.predict(test_feature_3))




