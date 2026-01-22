# 深度学习建模的框架
import torch

# 矩阵运算
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算

# 画图
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
# X_numpy = np.random.rand(100, 1) * 10
# # 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。
#
# y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)
# X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
# y = torch.from_numpy(y_numpy).float()

X_numpy = np.random.rand(100, 1) * 2 * np.pi
y_numpy = np.sin(X_numpy) + 0.05 * np.random.randn(100, 1)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# # 2. 直接创建参数张量 a 和 b
# # torch.randn() 生成随机值作为初始值。
# # y = a * x + b
# # requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
# a = torch.randn(1, requires_grad=True, dtype=torch.float)
# b = torch.randn(1, requires_grad=True, dtype=torch.float)
#
# print(f"初始参数 a: {a.item():.4f}")
# print(f"初始参数 b: {b.item():.4f}")
# print("---" * 10)

model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1)
)
print("---" * 10)
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
# a * x + b 《 - 》  y'

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 1
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    # y_pred = a * X + b
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
# a_learned = a.item()
# b_learned = b.item()
# print(f"拟合的斜率 a: {a_learned:.4f}")
# print(f"拟合的截距 b: {b_learned:.4f}")
print("---" * 10)

# 6. 绘制结果
# # 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
# with torch.no_grad():
#     y_predicted = a_learned * X + b_learned
#
# plt.figure(figsize=(10, 6))
# plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
# plt.plot(X_numpy, y_predicted, label=f'Model: y = {a_learned:.2f}x + {b_learned:.2f}', color='red', linewidth=2)
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()

x_grid = np.linspace(0, 2*np.pi, 400).reshape(-1, 1).astype(np.float32)
x_grid_t = torch.from_numpy(x_grid)

with torch.no_grad():
    y_pred_grid = model(x_grid_t).numpy()

y_true_grid = np.sin(x_grid)#新增

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
