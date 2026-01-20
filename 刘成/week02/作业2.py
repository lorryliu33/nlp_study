# 2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。


import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.optim as optim
# 1. 生成模拟数据 (与之前相同)
torch.manual_seed(42)
np.random.seed(42)



x = np.linspace(0, 2 * np.pi, 1000, dtype=np.float32)
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y = np.sin(x) + 0.05 * np.random.randn(*x.shape)
x_tensor = torch.from_numpy(x).unsqueeze(1).float()  # 形状：(1000, 1)
y_tensor = torch.from_numpy(y).unsqueeze(1).float()  # 形状：(1000, 1)

print("数据生成完成。")
print("---" * 10)

# 2. 构建多层网络架构

class SinFittingNet(nn.Module):
    def __init__(self):
        super(SinFittingNet, self).__init__()
        # 定义多层全连接层：输入层(1) -> 隐藏层1(64) -> 隐藏层2(64) -> 隐藏层3(32) -> 输出层(1)
        self.layers = nn.Sequential(
            nn.Linear(1, 64),    
            nn.ReLU(),          
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),   
            nn.ReLU(),
            nn.Linear(32, 1)     
        )
    
    def forward(self, x):
        # 前向传播
        return self.layers(x)







# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
model = SinFittingNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
loss_fn = torch.nn.MSELoss() # 回归任务

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。


# 4. 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    outputs = model(x_tensor)

    # 计算损失
    loss = loss_fn(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")


print("---" * 10)
y_pred = model(x_tensor).detach().numpy()
# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

# 绘制sin函数和拟合结果
ax1.plot(x, y, label='Original sin(x) (with noise)', color='blue', alpha=0.5)
ax1.plot(x, y_pred, label='NN Fitted Curve', color='red', linewidth=2)
ax1.set_title('Sin Function Fitting with Multi-Layer Network')
ax1.set_xlabel('x')
ax1.set_ylabel('sin(x)')
ax1.legend()
ax1.grid(True)

# 调整布局并显示
plt.tight_layout()
plt.show()