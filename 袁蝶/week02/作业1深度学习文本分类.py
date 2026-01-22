import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # 用于可视化loss对比

# ===================== 1. 数据加载与预处理 =====================
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签转数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符转索引
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40


# ===================== 2. 数据集类 =====================
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


# ===================== 3. model =====================
class ConfigurableClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        可配置层数的分类器
        :param input_dim: 输入维度（vocab_size）
        :param hidden_dims: 隐藏层维度列表，比如 [64] 表示1层隐藏层（64节点），[128, 64] 表示2层隐藏层（128→64）
        :param output_dim: 输出维度（类别数）
        """
        super().__init__()
        layers = []
        # 第一层：输入→第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        # 中间隐藏层：前一个隐藏层→后一个隐藏层
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())

        # 输出层：最后一个隐藏层→输出
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # 把所有层组合成Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# hidden_dim = 128
# output_dim = len(label_to_index)
# model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
# criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# ===================== 4. 训练函数 =====================
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    """
    训练模型并返回每轮的loss
    :return: 每轮epoch的平均loss列表
    """
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算本轮平均loss
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
    return epoch_losses


# ===================== 5. 预测函数 =====================
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
    predicted_label = index_to_label[predicted_index.item()]
    return predicted_label


# ===================== 6. 对比实验 =====================
if __name__ == "__main__":
    # 初始化数据集和dataloader
    char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)# 读取批量数据集 -》 batch数据
    output_dim = len(label_to_index)
    num_epochs = 10  # 训练轮数
    lr = 0.01  # 学习率

    # 定义要对比的模型结构（层数+节点数）
    # 格式：{模型名称: 隐藏层维度列表}
    model_configs = {
        "1层-64节点": [64],  # 基础：1层隐藏层，64个节点
        "1层-128节点": [128],  # 对比：节点
        "1层-256节点": [256],  # 对比：节点
        "2层-128→64节点": [128, 64],  # 对比：层数2层
        "3层-256→128→64节点": [256, 128, 64]  # 对比：3层
    }

    # 存储每个模型的loss记录
    all_losses = {}

    # 逐个训练模型并记录loss
    for model_name, hidden_dims in model_configs.items():
        print("=" * 50)
        print(f"开始训练：{model_name}")
        # 初始化模型、损失函数、优化器
        model = ConfigurableClassifier(vocab_size, hidden_dims, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # 训练并记录loss
        losses = train_model(model, dataloader, criterion, optimizer, num_epochs)
        all_losses[model_name] = losses

    # ===================== 7. 可视化loss对比 =====================
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False

    for model_name, losses in all_losses.items():
        plt.plot(range(1, num_epochs + 1), losses, label=model_name)

    plt.xlabel("Epoch（训练轮数）")
    plt.ylabel("Average Loss（平均损失）")
    plt.title("不同模型结构的Loss变化对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("model_loss_comparison.png")  # 保存图片
    plt.show()

    # ===================== 8. 用最优模型（loss最低的）做预测 =====================
    # 找到loss最低的模型
    best_model_name = min(all_losses.keys(), key=lambda x: all_losses[x][-1])
    best_hidden_dims = model_configs[best_model_name]
    best_model = ConfigurableClassifier(vocab_size, best_hidden_dims, output_dim)
    # 重新训练最优模型
    optimizer = optim.SGD(best_model.parameters(), lr=lr)
    train_model(best_model, dataloader, criterion, optimizer, num_epochs)

    # 预测示例
    index_to_label = {i: label for label, i in label_to_index.items()}
    new_texts = ["帮我导航到北京", "查询明天北京的天气"]
    for text in new_texts:
        pred = classify_text(text, best_model, char_to_index, vocab_size, max_len, index_to_label)
        print(f"输入 '{text}' 预测为: '{pred}'")
