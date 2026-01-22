
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}  #去重，枚举，映射
numerical_labels = [label_to_index[label] for label in string_labels] #把每条样本的字符串标签替换成字符标签

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

# 新增的 自定义数据集
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_count=1): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        '''self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)'''
        layers = []
        in_dim = input_dim

        # 堆叠 num_layers 个隐藏层：Linear + ReLU
        for _ in range(layer_count):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # 输出层（logits）
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        # 手动实现每层的计算
        '''out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out'''

char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

#初始化训练要素：模型，损失，优化器
'''hidden_dims = [64, 128, 256]#1111
for hidden_dim in hidden_dims:
##hidden_dim = 128
print(f"\n===== hidden_dim = {hidden_dim} =====")#2222
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.01)'''

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次
###训练循环
output_dim = len(label_to_index)
num_epochs = 10
lr = 0.01

configs = [
    (1, 64),
    (1, 128),
    (2, 128),
    (3, 128),
    (2, 256),
]

all_results = {}

for layer_count, hidden_dim in configs:
    print(f"layer_count:{layer_count}, hidden_dim:{hidden_dim}")
    print("\n")

    model = SimpleClassifier(vocab_size, hidden_dim, output_dim, layer_count=layer_count)
    criterion = nn.CrossEntropyLoss() # 损失函数 内部自带 log-softmax + NLLLoss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    epoch_losses = []

    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        '''print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")'''

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    all_results[f"layers={layer_count},hidden={hidden_dim}"] = epoch_losses




# 训练完后，简单打印“每组最终loss”
print("\n\n===== Final Avg Loss Comparison =====")
for name, losses in all_results.items():
    print(f"{name:22s} -> final avg loss: {losses[-1]:.4f}")


##推理函数：输入一句话，输出预测标签
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "5月8号下午4点之前的提醒取消掉"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "交通类的武汉交通广播电台来一个吧"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")