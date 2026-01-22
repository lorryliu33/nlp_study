import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import csv

df = pd.read_csv('dataset_1.csv', sep='\t', header=None, names=['text', 'label'])

def chinese_tokenizer(text):
    return jieba.lcut(text.strip(), cut_all=False)
print(df.head())
def simple_tokenize(text):
    return ' '.join(jieba.lcut(str(text)))

df['text_processed'] = df['text'].apply(simple_tokenize)

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['text_processed'])

le = LabelEncoder()
y = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("=== SVM模型 ===")
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(f"SVM准确率: {accuracy_score(y_test, y_pred_svm):.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

print("\n=== 随机森林模型 ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"随机森林准确率: {accuracy_score(y_test, y_pred_rf):.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

def predict_new_text(text):
    processed = simple_tokenize(text)
    vectorized = vectorizer.transform([processed])
    svm_pred = svm.predict(vectorized)[0]
    rf_pred = rf.predict(vectorized)[0]
    return le.inverse_transform([svm_pred])[0], le.inverse_transform([rf_pred])[0]

test_texts = ["程度车标", "查询车票", "观看视频"]
for text in test_texts:
    svm_label, rf_label = predict_new_text(text)
    print(f"文本: '{text}' -> SVM预测: {svm_label}, 随机森林预测: {rf_label}")
