import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv('dataset.csv')

# 准备标签和训练数据
labels = {'正面': 0, '负面': 1}
df['label_id'] = df['label'].map(labels)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_id'], test_size=0.3, random_state=42
)

# 初始化BERT
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 编码数据
def encode_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )

train_encodings = encode_texts(X_train)
test_encodings = encode_texts(X_test)

# 创建数据集
train_labels = torch.tensor(y_train.values)
test_labels = torch.tensor(y_test.values)

# 初始化模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
).to(device)

# 简单训练
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("训练BERT模型...")
for epoch in range(2):
    model.train()
    
    # 单批次训练
    input_ids = train_encodings['input_ids'].to(device)
    attention_mask = train_encodings['attention_mask'].to(device)
    labels = train_labels.to(device)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # 评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(
            input_ids=test_encodings['input_ids'].to(device),
            attention_mask=test_encodings['attention_mask'].to(device)
        )
        predictions = torch.argmax(test_outputs.logits, dim=1)
        accuracy = (predictions == test_labels.to(device)).float().mean()
    
    print(f"Epoch {epoch+1}, 准确率: {accuracy.item():.4f}")
  
# 预测
model.eval()
test_text = "肖战演技太棒了"
encoding = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True, max_length=64)

with torch.no_grad():
    outputs = model(
        input_ids=encoding['input_ids'].to(device),
        attention_mask=encoding['attention_mask'].to(device)
    )
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"\n测试评论: '{test_text}'")
print(f"预测结果: {'正面' if prediction == 0 else '负面'}")
