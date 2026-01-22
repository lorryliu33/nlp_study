"""
文本分类程序
使用scikit-learn进行文本分类任务
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 数据集路径
dataset_path = 'dataset.csv'


def load_data(file_path):
    """
    加载数据集
    
    Args:
        file_path (str): 数据集文件路径
        
    Returns:
        tuple: (文本列表, 标签列表)
    """
    print(f"正在加载数据: {file_path}")
    
    # 读取CSV文件，文本\t标签
    df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
    
    # 提取文本和标签
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(str).tolist()
    
    print(f"数据加载完成，共 {len(texts)} 条样本")
    print(f"类别数量: {len(set(labels))}")
    print(f"类别分布:\n{df['label'].value_counts()}\n")
    
    return texts, labels


def preprocess_data(texts, labels, test_size=0.2, random_state=42):
    """
    数据预处理和划分
    
    Args:
        texts (list): 文本列表
        labels (list): 标签列表
        test_size (float): 测试集比例
        random_state (int): 随机种子
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("正在进行数据预处理...")
    
    # 使用TF-IDF向量化文本（比CountVectorizer重点好在可以提高稀有词的权重）
    # max_features: 限制特征数量，提高效率
    # ngram_range: 使用1-gram和2-gram
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # 将文本转换为TF-IDF特征矩阵
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    
    print(f"特征维度: {X.shape}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # 保持类别分布
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}\n")
    
    return X_train, X_test, y_train, y_test, vectorizer


def train_model(X_train, y_train, model_type='lr'):
    """
    训练分类模型
    
    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        model_type (str): 模型类型 ('lr', 'nb', 'svm')
        
    Returns:
        训练好的模型
    """
    print(f"正在训练 {model_type} 模型...")

    match model_type:
        case 'lr':
            # 逻辑回归：速度快，效果好，适合文本分类
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs',
                multi_class='multinomial'
            )
        case 'nb':
            # 朴素贝叶斯：适合文本分类，速度快
            model = MultinomialNB(alpha=1.0)
        case 'svm':
            # 支持向量机：效果好但速度较慢
            model = SVC(kernel='linear', random_state=42, probability=True)
        case _:
            # 默认情况：不支持的模型类型
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    model.fit(X_train, y_train)
    print("模型训练完成\n")
    
    return model

def predict_text(model, vectorizer, text):
    """
    对单个文本进行预测
    
    Args:
        model: 训练好的模型
        vectorizer: 文本向量化器
        text (str): 待预测的文本
        
    Returns:
        tuple: (预测类别, 置信度)
    """
    # 向量化文本
    text_vector = vectorizer.transform([text])
    
    # 预测类别
    prediction = model.predict(text_vector)[0]
    
    return prediction


def main():
    """
    主函数：执行完整的文本分类流程
    """
    # 1. 加载数据
    texts, labels = load_data(dataset_path)
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(texts, labels)
    
    # 3. 训练模型（可以选择不同的模型）
    # 可选: 'lr' (逻辑回归), 'nb' (朴素贝叶斯), 'svm' (支持向量机)
    model = train_model(X_train, y_train, model_type='nb')
    
    # 4. 示例：预测新文本
    print("=" * 50)
    print("预测示例:")
    test_texts = [
        "播放一首周杰伦的歌",
        "查询北京到上海的航班",
        "打开空调",
        "我喜欢打游戏"
    ]
    
    for text in test_texts:
        pred = predict_text(model, vectorizer, text)
        print(f"文本: {text}")
        print(f"预测类别: {pred}\n")
    
    print("=" * 50)
    print("文本分类任务完成！")


if __name__ == '__main__':
    main()
