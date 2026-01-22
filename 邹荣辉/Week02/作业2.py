import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成模拟数据
X_numpy = np.random.rand(5000, 1) * 4 * np.pi
y_numpy = np.sin(X_numpy) + np.random.randn(5000, 1) * 0.1

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


# 2. 定义多层神经网络模型
class SinFittingModel(nn.Module):
    def __init__(self):
        super(SinFittingModel, self).__init__()
        # 四层神经网络
        self.fc1 = nn.Linear(1, 128)
        self.tanh1 = nn.Tanh()# Tanh激活函数更适合拟合sin
        self.fc2 = nn.Linear(128, 128)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(128, 64)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.fc3(x)
        x = self.tanh3(x)
        x = self.fc4(x)
        return x

# 创建模型实例
model = SinFittingModel()

print("模型初始化完成。")
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  #使用Adam优化器，更适合拟合非线性函数,学习率0.001

# 4. 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播：使用模型
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每500个epoch打印一次损失
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 打印最终损失
print("\n训练完成！")
print(f"最终损失: {loss.item():.6f}")
print("---" * 10)

# 6. 绘制结果
X_plot = torch.linspace(0, 4 * np.pi, 1000).reshape(-1, 1).float()
with torch.no_grad():
    y_pre = model(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_plot, y_pre, label=f'Model: y =sin(x))', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
