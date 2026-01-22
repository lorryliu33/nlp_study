import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. 读取数据
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print("数据集前5行:")
print(dataset.head(5))
print(f"\n数据集形状: {dataset.shape}")
print(f"类别分布:\n{dataset[1].value_counts()}")

# 2. 数据预处理 - 中文分词
print("\n正在进行中文分词...")
dataset[0] = dataset[0].apply(lambda x: " ".join(jieba.lcut(str(x))))

# 3. 特征提取 - 使用TF-IDF
print("正在提取TF-IDF特征...")
vectorizer = TfidfVectorizer(max_features=1000)  # 限制特征数量
X = vectorizer.fit_transform(dataset[0].values)
y = dataset[1].values

# 4. 划分训练集和测试集
print("划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 5. 定义多个分类器
classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    '朴素贝叶斯': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=True),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 6. 训练和评估所有模型
results = {}
print("\n" + "="*60)
print("开始训练和评估模型...")
print("="*60)

for name, clf in classifiers.items():
    print(f"\n训练 {name} 模型...")
    clf.fit(X_train, y_train)
    
    # 在测试集上预测
    y_pred = clf.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': clf,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"{name} 准确率: {accuracy:.4f}")
    print(f"分类报告:\n{classification_report(y_test, y_pred)}")

# 7. 比较所有模型性能
print("\n" + "="*60)
print("模型性能比较:")
print("="*60)
for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name}: 准确率 = {result['accuracy']:.4f}")

# 8. 保存最佳模型
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
print(f"\n最佳模型: {best_model_name}")

