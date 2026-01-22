import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv('dataset.csv')

# 准备标签
label_map = {'正面': 0, '负面': 1}
df['label_id'] = df['label'].map(label_map)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_id'], test_size=0.3, random_state=42
)

# 初始化BERT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 创建数据集类
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 创建数据加载器
train_dataset = CommentDataset(X_train, y_train, tokenizer)
test_dataset = CommentDataset(X_test, y_test, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# 初始化模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
).to(device)

# 训练配置
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 训练函数
def train_epoch(model, data_loader, optimizer):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# 评估函数
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)_, preds = torch.max(outputs.logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

# 训练模型
print("开始训练BERT模型...")
for epoch in range(2):
    train_epoch(model, train_loader, optimizer)
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}, 准确率: {acc:.4f}")

# 测试示例
model.eval()
test_text = "肖战演技太棒了，剧情扣人心弦"
encoding = tokenizer.encode_plus(
    test_text,
    max_length=64,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

with torch.no_grad():
    outputs = model(
        input_ids=encoding['input_ids'].to(device),
        attention_mask=encoding['attention_mask'].to(device)
    )
    prediction = torch.argmax(outputs.logits, dim=1).item()

result = '正面' if prediction == 0 else '负面'
print(f"\n测试评论: '{test_text}'")
print(f"BERT预测结果: {result}")
