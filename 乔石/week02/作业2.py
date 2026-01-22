import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(42)

# 1、使用numpy创建数据
X_np = np.random.rand(500, 1) * 2 * np.pi
y_np = np.sin(X_np) + np.random.normal(0, 0.1, size=X_np.shape)

# 2、将numpy数据转换成torch张量
X = torch.from_numpy(X_np).float()
y = torch.from_numpy(y_np).float()


# 3、定义多层网络模型
class SinModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(SinModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, output_size)
        )

    def forward(self, x):
        return self.network(x)


# --- 模型参数和实例化 ---
input_size = 1
hidden_size1 = 60
hidden_size2 = 120
hidden_size3 = 26
output_size = 1
model = SinModel(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# 4、定义损失函数和优化器
loss_fn = torch.nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

# 5、训练模型
num_epochs = 1000
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
# 6、用模型进行预测
model.eval()
with torch.no_grad():
    y_pred = model(X)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(X_np, y_np, label='Raw data', color='blue', alpha=0.6)
plt.scatter(X_np, y_pred.numpy(), label=f'Model', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
