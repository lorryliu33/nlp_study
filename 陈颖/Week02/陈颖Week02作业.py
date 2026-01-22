作业一代码：
import pandas as pd
import torch
import torch.nn as nn #神经网络层与损失函数
import torch.optim as optim #优化器 SGD/Adam
from torch.utils.data import Dataset, DataLoader#数据集封装与批量加载

#读取数据  文本+标签
# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
#第0列是文本，把这一列转化成为pythonlist
string_labels = dataset[1].tolist()
#第一列是标签，也转变为list

##标签编码： string to index
label_to_index = {label: i for i, label in enumerate(set(string_labels))}  #去重，枚举，映射
numerical_labels = [label_to_index[label] for label in string_labels] #把每条样本的字符串标签替换成字符标签

#字符表构建： char - index
char_to_index = {'<pad>': 0}
## 预留一个padding字符 <pad> 的 id=0；用于补齐长度、也用于“未知字符”
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
#遇到新字符就分配一个新id

index_to_char = {i: char for char, i in char_to_index.items()}
#反向字典
vocab_size = len(char_to_index)
#字符表大小

max_len = 40

#定义Dataset：把文本转成Bow向量
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


#定义分类模型，两层全连接
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,num_layers=1): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        '''self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)'''
        layers = []
        in_dim = input_dim

        # 堆叠 num_layers 个隐藏层：Linear + ReLU
        for _ in range(num_layers):
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

#构建DataLoader:按batch喂给模型
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

#新增部分
# 最小改动 2：多组配置对比 loss
# =======================
# (num_layers, hidden_dim)
configs = [
    (1, 64),
    (1, 128),
    (2, 128),
    (3, 128),
    (2, 256),
]

all_results = {}  # 保存每组配置的 epoch loss，用于你后续想打印/画图

for num_layers, hidden_dim in configs:
    print("\n" + "=" * 60)
    print(f"Config: num_layers={num_layers}, hidden_dim={hidden_dim}")
    print("=" * 60)

    model = SimpleClassifier(vocab_size, hidden_dim, output_dim, num_layers=num_layers)
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

        # 新增部分
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    all_results[f"layers={num_layers},hidden={hidden_dim}"] = epoch_losses

    # epoch： 将数据集整体迭代训练一次
    # batch： 数据集汇总为一批训练一次




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

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")


作业二代码：
import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
'''X_numpy = np.random.rand(100, 1) * 10
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()'''
X_numpy = np.random.rand(100, 1) * 2 * np.pi   # x in [0, 2π)
y_numpy = np.sin(X_numpy)                       # y = sin(x)

y_numpy = y_numpy + 0.05 * np.random.randn(100, 1)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()
print("数据生成完成。")
print("---" * 10)

''''# 2. 直接创建参数张量 a 和 b
# torch.randn() 生成随机值作为初始值。
# y = a * x + b
# requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
print("---" * 10)'''

# 2. 用多层网络替换 a,b（核心改动）
# 原来：a = ... , b = ...
# 现在：用 MLP 逼近 sin
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1)
)

print("模型创建完成：多层网络拟合 sin(x)")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
# a * x + b 《 - 》  y'

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.SGD(model.parameters(), lr=0.065) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    '''y_pred = a * X + b'''
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
'''a_learned = a.item()
b_learned = b.item()
print(f"拟合的斜率 a: {a_learned:.4f}")
print(f"拟合的截距 b: {b_learned:.4f}")'''
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
##新增
# 为了画“连续曲线”，做一个更密集的 x_grid
x_grid = np.linspace(0, 2*np.pi, 400).reshape(-1, 1).astype(np.float32)
x_grid_t = torch.from_numpy(x_grid)

with torch.no_grad():
    '''y_predicted = a_learned * X + b_learned'''
    y_pred_grid = model(x_grid_t).numpy()

y_true_grid = np.sin(x_grid)#新增

'''plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model: y = {a_learned:.2f}x + {b_learned:.2f}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()'''

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Train samples (noisy)', alpha=0.6)
plt.plot(x_grid, y_true_grid, label='True: sin(x)', linewidth=2)
plt.plot(x_grid, y_pred_grid, label='NN prediction', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('MLP Fit sin(x)')
plt.legend()
plt.grid(True)
plt.show()
