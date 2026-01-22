import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 定义不同结构的模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()

        # 创建隐藏层
        prev_dim = input_dim
        for hidden_dim in layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # 输出层
        self.output = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


# 模型配置（层数和节点数）
configs = [
    {"name": "1层(64)", "layers": [64]},
    {"name": "1层(128)", "layers": [128]},
    {"name": "1层(256)", "layers": [256]},
    {"name": "2层(128,64)", "layers": [128, 64]},
    {"name": "3层(256,128,64)", "layers": [256, 128, 64]},
]

# 假设已有数据
vocab_size = 100  # 示例
num_classes = 5
batch_size = 32

# 模拟训练
epochs = 10
results = []

for config in configs:
    print(f"训练:{config['name']}")
    model = SimpleClassifier(vocab_size, config['layers'], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    for epoch in range(epochs):
        # 训练循环
        optimizer.zero_grad()
        # 这里应该有实际数据，这里用随机数据模拟
        inputs = torch.randn(batch_size, vocab_size)
        labels = torch.randint(0, num_classes, (batch_size,))
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    results.append({"name": config['name'], "losses": losses})

# 可视化
plt.figure(figsize=(10, 5))
for result in results:
    plt.plot(result['losses'], label=result['name'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同网络结构的Loss对比')
plt.legend()
plt.grid(True)
plt.show()
