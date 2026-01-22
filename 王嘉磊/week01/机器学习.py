import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer   # 词频统计
from sklearn.neighbors import KNeighborsClassifier   # KNN
from openai import OpenAI
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 进行转换 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
client = OpenAI(
   api_key="sk-4cdef86c739746f9b6bac72ecfd213f6",
  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_ml(text: str) -> str:
  test_sentence = " ".join(jieba.lcut(text))
  test_feature = vector.transform([test_sentence])
  return model.predict(test_feature)[0]
if __name__ == "__main__":
 print("2222")
 print("机器学习: ", text_calssify_using_ml("帮我导航到天安门"))
