import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset

#1.读取文件
dateset = pd.read_csv("../Week01/dataset.csv", sep='\t', header=None)
# 数据分为两列
text_column = dateset[0]
label_column = dateset[1]
#2.创建标签到索引的映射关系 结果应为：{'Radio-Listen': 0, 'Calendar-Query': 1} 给每个类型变成了一个索引
# label_to_index = {}
# for index, label in enumerate(set(label_column)):
#     label_to_index[label] = index
label_to_index = {label: index for index, label in enumerate(set(label_column))}

#3.创建索引到标签的映射关系 结果是：[10,10] 把每一行分类对应的索引转换成label_to_index的对应索引
# numerical_labels = []
# for label in label_column:
#     numerical_labels.append(label_to_index[label])
numerical_labels = [label_to_index[label] for label in label_column]


#4.创建字符到索引的映射关系表 ，把每个字符（去重），都变成一个索引
char_to_index = {'<pad>': 0}
for text in text_column:
    for char in text:
        if char not in char_to_index:
            # 字符索引
            char_to_index[char] = len(char_to_index)

#5.创建索引到字符的映射关系表, 把结果4的内容转换一下
index_to_char = {i: char for char, i in char_to_index.items()}

#字典表长度
vocab_size = len(char_to_index)
#认为数据中前40个字符足够表达含义
max_len = 40

#6.简单的神经网络模型
class SimpleClassifier(nn.Module):
    # 网络深度: 当前是2层全连接网络结构
    # input_dim 输入维度 = 词汇表大小 hidden_dim 隐藏层维度 = 128 output_dim 输出维度 = 分类数量
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)  # 新增隐藏层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)  # 新增隐藏层
        self.fc3 = nn.Linear(hidden_dim//2, output_dim)  # 第三层
        self.relu = nn.ReLU()
        # dropout 概率，表示神经元被随机置零的概率 防止过拟合 通常三层以上网络结构，需要
        # 输入层: 0.1~0.2（保留更多信息）
        # 隐藏层: 0.3~0.5（平衡效果）
        # 靠近输出层: 0.5~0.7（更强正则化）
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)  # 修正：fc1后添加激活
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)  # 修正：fc2后添加激活
        out = self.dropout(out)
        out = self.fc3(out)
        return out

#7.文本数据转换为pytorch类
class CharBoWDataset(Dataset):
    # texts，原始文本数据 labels: 标签列表，对应的标签数据 char_to_index: 字符到索引的映射字典max_len: 最大文本长度，用于统一序列长度
    # vocab_size: 词汇表大小，词袋向量维度
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long) # 标签内容转换为张量(torch需要的数据) 类型为long
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    # 创建词袋向量
    def _create_bow_vectors(self):
        tokenized_texts = []
        # 把内容文本，取最大字符进行拼接，不足的填充0
        for text in self.texts:
            #截取前40个字符
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 不足的填充0
            tokenized += [0] * (self.max_len - len(tokenized))
            # 拼接内容
            tokenized_texts.append(tokenized)

        # 文本转换为数值向量
        bow_vectors = []
        for text_indices in tokenized_texts:
            # 创建一个全零向量，维度为词汇表大小
            bow_vector = torch.zeros(self.vocab_size)
            # 遍历文本内容，把每个字符出现的次数，保存在bow_vector中
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


#8.创建数据集
char_dataset = CharBoWDataset(text_column, numerical_labels, char_to_index, max_len, vocab_size)

labels_tensor = torch.tensor(numerical_labels, dtype=torch.long)
tensor_dataset = TensorDataset(char_dataset.bow_vectors, labels_tensor)
# 将数据集加载成批量数据
# 32 一批的数量，常见1-128 主要看性能和内存 shuffle=True 打乱顺序，防拟合
dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

#9.创建模型
hidden_dim = 128
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.01) # Adam 优化器  可以结合梯度 动态调整学习， 0.01 -> 0.001 -> 0.00001

#10.训练模型
# epoch： 将数据集整体迭代训练一次
num_epochs = 100
for epoch in range(num_epochs):
    # 启用训练模式
    model.train()
    # 初始化累计损失值为0，用来记录这一轮训练的整体损失
    running_loss = 0.0

    # 开始遍历数据加载器，每次取出一个批次的输入数据和对应标签
    for inputs, labels in dataloader:
        # 把优化器里的梯度清零，准备计算新一轮的梯度
        optimizer.zero_grad()
        # 把输入数据送进模型，得到模型的预测输出
        outputs = model(inputs)
        # 计算模型输出和真实标签之间的差距（损失值）
        loss = criterion(outputs, labels)
        # 反向传播：计算损失相对于每个参数的梯度
        loss.backward()
        # 根据计算出的梯度更新模型参数
        optimizer.step()
        # 把当前批次的损失加到总损失里，用于记录整体训练进度
        running_loss += loss.item()
        # 每50个批次就打印一次当前的损失值，让我们知道训练进展
    #     if idx % 50 == 0:
    #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

#11.测试模型
#推理函数字段说明：
# text: 输入文本，
# model: 训练好的模型，
# char_to_index: 字符到索引的映射字典，
# vocab_size: 词汇表大小，
# max_len: 输入文本的最大长度，
# index_to_label: 索引到标签的映射字典
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将输入文本转换为字符索引序列，使用char_to_index字典映射
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 如果文本长度不足max_len，则用0填充到固定长度
    tokenized += [0] * (max_len - len(tokenized))
    # 创建一个零向量，用于构建词袋向量
    bow_vector = torch.zeros(vocab_size)
    # 统计每个字符在文本中的出现次数，构建词袋向量
    for index in tokenized:
        if index != 0:  # 跳过填充字符（0）
            bow_vector[index] += 1
    # 增加一个维度，使其成为批次格式（模型期望批次输入）
    bow_vector = bow_vector.unsqueeze(0)
    # 切换模型到评估模式（关闭训练特有的功能如dropout）
    model.eval()
    # 禁用梯度计算，节省内存并加快推理速度
    with torch.no_grad():
        # 将词袋向量输入模型，获取预测输出
        output = model(bow_vector)
    # 找到输出中概率最大的类别索引
    _, predicted_index = torch.max(output, 1)
    # 将张量转换为Python整数
    predicted_index = predicted_index.item()
    # 使用index_to_label字典将索引转换为实际的标签名称
    predicted_label = index_to_label[predicted_index]
    # 返回预测的类别标签
    return predicted_label

# 创建索引到标签的映射字典，用于将数字索引转换回原始标签名称
index_to_label = {i: label for label, i in label_to_index.items()}

#执行预测
new_text = "我不开心"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text = "起飞"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")