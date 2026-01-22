import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以获得可重现的结果
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成模拟数据
X_numpy = np.linspace(0, 4 * np.pi, 1000).reshape(-1, 1)  # 更密集的采样点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # 添加少量噪声

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("---" * 10)


# 2. 定义多层神经网络
class SineFitter(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[64, 32, 16], output_size=1):
        super(SineFitter, self).__init__()

        layers = []
        prev_size = input_size

        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # 使用ReLU激活函数增加非线性
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 创建网络实例
model = SineFitter(input_size=1, hidden_sizes=[128, 64, 32], output_size=1)

print(f"模型结构:\n{model}")
print("---" * 10)

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 4. 训练模型
num_epochs = 2000
losses = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    # print("y_pred shape:", y_pred.shape) # y_pred shape: torch.Size([1000, 1])

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 评估模型
model.eval()  # 切换到评估模式
with torch.no_grad():
    y_predicted = model(X).numpy()
# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 6. 可视化结果
plt.figure(figsize=(15, 5))

# 子图1: 显示原始数据和拟合结果
plt.subplot(1, 3, 1)
plt.scatter(X_numpy[::10], y_numpy[::10], label='原始数据(带噪声)', color='blue', alpha=0.6, s=10)  # 每10个点取一个以减少重叠
plt.plot(X_numpy, np.sin(X_numpy), 'g--', label='真实sin函数', linewidth=2)
plt.plot(X_numpy, y_predicted, 'r-', label='神经网络拟合', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('多层神经网络拟合sin函数')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: 显示损失变化曲线
plt.subplot(1, 3, 2)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失变化')
plt.grid(True, alpha=0.3)

# 子图3: 显示预测值与真实值的对比
plt.subplot(1, 3, 3)
plt.plot(y_numpy, y_predicted, 'bo', alpha=0.5)
plt.plot([y_numpy.min(), y_numpy.max()], [y_numpy.min(), y_numpy.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测值 vs 真实值')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 额外测试：在训练范围外进行预测
X_test = np.linspace(-2 * np.pi, 6 * np.pi, 1000).reshape(-1, 1)
y_true_test = np.sin(X_test)
X_test_tensor = torch.from_numpy(X_test).float()

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy()

plt.figure(figsize=(12, 6))
plt.plot(X_test, y_true_test, 'g--', label='真实sin函数', linewidth=2)
plt.plot(X_test, y_pred_test, 'r-', label='神经网络拟合', linewidth=2)
plt.scatter(X_numpy[::20], y_numpy[::20], label='训练数据', color='blue', alpha=0.6, s=20)
plt.xlabel('X')
plt.ylabel('y')
plt.title('多层神经网络对sin函数的泛化能力（包括训练范围外）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"最终训练损失: {losses[-1]:.6f}")
