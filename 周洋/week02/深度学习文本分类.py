import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

# 数据加载和预处理
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40


# 数据集类（保持不变）
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


# 1. 原始模型：2层（128个隐藏节点）
class SimpleClassifier1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 2. 更简单的模型：2层（64个隐藏节点）
class SimpleClassifier2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 3. 更复杂的模型：3层（256-128个节点）
class DeeperClassifier1(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(DeeperClassifier1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 添加dropout防止过拟合

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out


# 4. 更宽更深的模型：4层（512-256-128个节点）
class DeeperClassifier2(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(DeeperClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc4(out)
        return out


# 5. 非常简单的模型：2层（32个隐藏节点）
class SmallClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SmallClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 训练函数
def train_model(model, model_name, dataloader, num_epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    times = []

    print(f"\n{'=' * 50}")
    print(f"训练模型: {model_name}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"{'=' * 50}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - epoch_start_time
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        times.append(epoch_time)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")

    return losses, times


# 主函数
def main():
    # 创建数据集和数据加载器
    char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

    output_dim = len(label_to_index)

    # 定义不同的模型配置
    model_configs = [
        {
            'name': '模型1: 2层(128节点)',
            'model': SimpleClassifier1(vocab_size, 128, output_dim),
            'color': 'blue'
        },
        {
            'name': '模型2: 2层(64节点)',
            'model': SimpleClassifier2(vocab_size, 64, output_dim),
            'color': 'green'
        },
        {
            'name': '模型3: 3层(256-128节点)',
            'model': DeeperClassifier1(vocab_size, 256, 128, output_dim),
            'color': 'red'
        },
        {
            'name': '模型4: 4层(512-256-128节点)',
            'model': DeeperClassifier2(vocab_size, 512, 256, 128, output_dim),
            'color': 'purple'
        },
        {
            'name': '模型5: 2层(32节点)',
            'model': SmallClassifier(vocab_size, 32, output_dim),
            'color': 'orange'
        }
    ]

    # 训练所有模型并收集结果
    all_results = {}

    for config in model_configs:
        model = config['model']
        model_name = config['name']

        losses, times = train_model(model, model_name, dataloader, num_epochs=10)

        all_results[model_name] = {
            'losses': losses,
            'times': times,
            'color': config['color'],
            'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'final_loss': losses[-1]
        }

    # 绘制损失曲线对比图
    plt.figure(figsize=(15, 10))

    # 损失曲线图
    plt.subplot(2, 2, 1)
    for model_name, results in all_results.items():
        plt.plot(results['losses'], label=model_name, color=results['color'], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('不同模型结构的Loss对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 最终损失对比图
    plt.subplot(2, 2, 2)
    model_names = list(all_results.keys())
    final_losses = [all_results[name]['final_loss'] for name in model_names]
    colors = [all_results[name]['color'] for name in model_names]

    bars = plt.bar(model_names, final_losses, color=colors, alpha=0.7)
    plt.xlabel('模型')
    plt.ylabel('最终Loss')
    plt.title('各模型最终Loss对比')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{loss:.4f}', ha='center', va='bottom')

    # 参数数量对比图
    plt.subplot(2, 2, 3)
    param_counts = [all_results[name]['params'] for name in model_names]
    bars = plt.bar(model_names, param_counts, color=colors, alpha=0.7)
    plt.xlabel('模型')
    plt.ylabel('参数数量')
    plt.title('模型参数数量对比')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # 训练时间对比图
    plt.subplot(2, 2, 4)
    avg_times = [sum(all_results[name]['times']) / len(all_results[name]['times']) for name in model_names]
    bars = plt.bar(model_names, avg_times, color=colors, alpha=0.7)
    plt.xlabel('模型')
    plt.ylabel('平均每轮训练时间(s)')
    plt.title('训练时间对比')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印详细对比表格
    print("\n" + "=" * 80)
    print("模型性能对比总结:")
    print("=" * 80)
    print(f"{'模型名称':<25} {'参数数量':<12} {'最终Loss':<12} {'平均训练时间(s)':<15}")
    print("-" * 80)

    for model_name in model_names:
        results = all_results[model_name]
        avg_time = sum(results['times']) / len(results['times'])
        print(f"{model_name:<25} {results['params']:<12} {results['final_loss']:<12.4f} {avg_time:<15.2f}")

    print("=" * 80)

    # 分析结果
    print("\n分析总结:")
    print("1. 参数数量与模型复杂度的关系:")
    for model_name in model_names:
        print(f"   {model_name}: {all_results[model_name]['params']} 个参数")

    print("\n2. Loss收敛情况:")
    best_model = min(all_results.items(), key=lambda x: x[1]['final_loss'])
    print(f"   最佳模型: {best_model[0]} (Loss: {best_model[1]['final_loss']:.4f})")

    print("\n3. 训练效率:")
    fastest_model = min(all_results.items(), key=lambda x: sum(x[1]['times']) / len(x[1]['times']))
    avg_time = sum(fastest_model[1]['times']) / len(fastest_model[1]['times'])
    print(f"   训练最快的模型: {fastest_model[0]} (平均每轮: {avg_time:.2f}s)")

if __name__ == "__main__":
    main()