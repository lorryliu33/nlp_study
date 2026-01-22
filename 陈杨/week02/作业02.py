import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数的训练数据
X_numpy = np.linspace(0, 4 * np.pi, 1000).reshape(-1, 1)  # 在0到4π之间生成1000个点
y_numpy = np.sin(X_numpy)  # sin函数值
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("---" * 10)


# 2. 定义多层神经网络来拟合sin函数
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),  # 输入层到第一个隐藏层
            nn.ReLU(),
            nn.Linear(64, 128),  # 第一个隐藏层到第二个隐藏层
            nn.ReLU(),
            nn.Linear(128, 64),  # 第二个隐藏层到第三个隐藏层
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出层
        )

    def forward(self, x):
        return self.network(x)


# 创建网络实例
model = SinNet()
print("网络结构:")
print(model)

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()  # MSE损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")

# 5. 生成测试数据进行预测
X_test_numpy = np.linspace(0, 6 * np.pi, 2000).reshape(-1, 1)  # 扩展到0到6π进行测试
X_test = torch.from_numpy(X_test_numpy).float()

# 使用训练好的模型进行预测
model.eval()  # 设置为评估模式
with torch.no_grad():  # 禁用梯度计算
    y_pred_test = model(X_test)
    y_pred_train = model(X)  # 也预测训练数据用于比较

train_loss = loss_fn(y_pred_train, y)
print(f"最终训练MSE: {train_loss.item():.6f}")


# 6. 可视化结果
plt.figure(figsize=(12, 8))
plt.plot(X_numpy, y_numpy, label='True sin(x)', color='blue', linewidth=2)
plt.plot(X_numpy, y_pred_train.numpy(), label='Fitted sin(x) - Train', color='red', linestyle='--')
plt.plot(X_test_numpy, y_pred_test.numpy(), label='Fitted sin(x) - Test', color='orange', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network Fitting of sin(x) Function')
plt.legend()
plt.grid(True, alpha=0.6)
plt.show()

