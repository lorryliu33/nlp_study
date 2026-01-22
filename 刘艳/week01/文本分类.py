import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# 文件名
file_path = 'dataset.csv'

try:
    print("正在加载数据...")

    # 修正读取方式：直接使用 sep='\t' (Tab分隔)，并指定列名为 text 和 label
    # on_bad_lines='skip' 会跳过那些格式不对（比如没有标签）的坏行，防止报错
    df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'], on_bad_lines='skip')

    # 去掉空值（以防万一）
    df.dropna(subset=['text', 'label'], inplace=True)

    print(f"成功加载数据，共 {len(df)} 条。")
    print("正在进行分词处理...")


    # 文本分词函数
    def chinese_tokenizer(text):
        if pd.isna(text):
            return ""
        # 转换成字符串再分词
        return " ".join(jieba.lcut(str(text)))


    df['text_cut'] = df['text'].apply(chinese_tokenizer)

    # 特征提取
    print("正在提取 TF-IDF 特征...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text_cut'])
    y = df['label']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型 1: 朴素贝叶斯
    print("\n---------- 模型 1: 朴素贝叶斯 (Naive Bayes) ----------")
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    y_pred_nb = nb_clf.predict(X_test)
    print(classification_report(y_test, y_pred_nb))
    print("准确率：", accuracy_score(y_test, y_pred_nb))

    # 模型 2: 支持向量机
    print("\n---------- 模型 2: 支持向量机 (SVM) ----------")
    svm_clf = LinearSVC(dual='auto')
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    print(classification_report(y_test, y_pred_svm))
    print("准确率：", accuracy_score(y_test, y_pred_svm))

except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请确保文件已拖入 PyCharm 项目左侧栏中。")
except Exception as e:
    print(f"程序运行出错: {e}")
