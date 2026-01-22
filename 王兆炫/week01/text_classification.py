import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. 加载数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 2. 中文分词处理
# 机器学习模型不认识中文句子，需要用 jieba 切词，并用空格连接
print("正在进行分词处理...")
def chinese_tokenizer(text):
    return " ".join(jieba.lcut(str(text)))

# 将第一列文本（dataset[0]）转换为分词后的格式
input_sentences = dataset[0].apply(chinese_tokenizer)
labels = dataset[1] # 第二列是类别标签

# 3. 特征提取 (向量化)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(input_sentences) # 将文本转为数字矩阵
y = labels

# 4. 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. 模型一：KNN (K-Nearest Neighbors)
# ==========================================
print("\n--- 正在训练模型 1: KNN ---")
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(train_x, train_y)
score_knn = model_knn.score(test_x, test_y)
print(f"KNN 模型在测试集上的准确率: {score_knn:.4f}")

# ==========================================
# 6. 模型二：逻辑回归 (Logistic Regression)
# ==========================================
print("\n--- 正在训练模型 2: 逻辑回归 ---")
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(train_x, train_y)
score_lr = model_lr.score(test_x, test_y)
print(f"逻辑回归模型在测试集上的准确率: {score_lr:.4f}")

# 7. 实操展示：输入一个例子进行预测
test_query = "帮我打开客厅的灯"
print(f"\n测试输入: '{test_query}'")

# 处理测试输入
test_sentence = chinese_tokenizer(test_query)
test_feature = vectorizer.transform([test_sentence])

# 预测结果
res_knn = model_knn.predict(test_feature)
res_lr = model_lr.predict(test_feature)

print(f"KNN 预测类别: {res_knn[0]}")
print(f"逻辑回归 预测类别: {res_lr[0]}")
