import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# =========================
# 1. 加载数据 & 预处理
# =========================
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)

texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40

print("数据加载完成")

# =========================
# 2. Dataset
# =========================
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab_size = vocab_size
        self.bow_vectors = self._build_bow(texts, char_to_index, max_len)

    def _build_bow(self, texts, char_to_index, max_len):
        vectors = []
        for text in texts:
            indices = [char_to_index.get(c, 0) for c in text[:max_len]]
            indices += [0] * (max_len - len(indices))

            bow = torch.zeros(self.vocab_size)
            for idx in indices:
                if idx != 0:
                    bow[idx] += 1
            vectors.append(bow)
        return torch.stack(vectors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

dataset = CharBoWDataset(
    texts, numerical_labels, char_to_index, max_len, vocab_size
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# 3. 通用 MLP 模型（支持任意层数）
# =========================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================
# 4. 训练函数
# =========================
def train_model(hidden_dims, num_epochs=20, lr=0.001):
    model = MLPClassifier(
        input_dim=vocab_size,
        hidden_dims=hidden_dims,
        output_dim=len(label_to_index)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

    return avg_loss, model

# =========================
# 5. 多模型结构对比
# =========================
model_configs = [
    [128, 64],
    [256, 128],
    [256, 128, 64],
    [256, 128, 64, 32],
    [512, 256],
    [512, 256, 128],
]

best_loss = float("inf")
best_model = None
best_config = None
results = []

for hidden_dims in model_configs:
    print(f"\n训练模型 hidden_dims={hidden_dims}")
    loss, model = train_model(hidden_dims)

    print(f"最终 Loss: {loss:.4f}")
    results.append((hidden_dims, loss))

    if loss < best_loss:
        best_loss = loss
        best_model = model
        best_config = hidden_dims

# =========================
# 6. 输出对比结果
# =========================
print("\n========== 模型对比结果 ==========")
for config, loss in results:
    print(f"hidden_dims={config}, loss={loss:.4f}")

print("\n最优模型")
print(f"hidden_dims={best_config}, best_loss={best_loss:.4f}")

# =========================
# 7. 推理函数
# =========================
def classify_text(text, model):
    indices = [char_to_index.get(c, 0) for c in text[:max_len]]
    indices += [0] * (max_len - len(indices))

    bow = torch.zeros(vocab_size)
    for idx in indices:
        if idx != 0:
            bow[idx] += 1

    bow = bow.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow)

    pred = torch.argmax(output, dim=1).item()
    return index_to_label[pred]

# =========================
# 8. 测试
# =========================
test_text = "帮我导航到北京"
predicted = classify_text(test_text, best_model)
print(f"\n输入：{test_text}")
print(f"预测结果：{predicted}")