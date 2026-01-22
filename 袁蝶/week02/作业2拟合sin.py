import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = ["SimHei"]

# 生成x：在[0, 2π]范围内生成400个均匀分布的点，形状(400, 1)
x_numpy = np.linspace(0, 2 * np.pi, 400).reshape(-1, 1)
# 生成y：sin(x) + 少量噪声，模拟真实场景
y_numpy = np.sin(x_numpy) + 0.1 * np.random.randn(*x_numpy.shape)

# 转换为PyTorch张量（float类型）
X = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成，x范围：[0, 2π]，共200个样本")
print("---" * 15)


# ===================== 定义多层神经网络 =====================
class SinFittingNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[128,64,32], output_dim=1):
        """
        拟合sin函数的多层全连接网络
        :param input_dim: 输入维度（x是一维的）
        :param hidden_dims: 隐藏层维度列表，[64,32]表示2层隐藏层（64→32）
        :param output_dim: 输出维度（y是一维的）
        """
        super().__init__()
        layers = []
        # 第一层：输入→第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())  # 非线性激活，让网络能拟合曲线

        # 中间隐藏层：前一层→后一层
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())

        # 输出层：最后一个隐藏层→输出（无激活，回归任务）
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # 组合所有层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ===================== 初始化模型、损失函数、优化器 =====================
# 初始化多层网络
model = SinFittingNet()
# 回归任务用均方误差损失
criterion = nn.MSELoss()
# 优化器（Adam比SGD收敛更快，适合拟合曲线）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # 优化器，基于 a b 梯度 自动更新

# ===================== 训练模型 =====================
num_epochs = 1000  # 训练轮数
loss_history = []  # 记录每轮loss
pred_history = []  # 记录关键轮次的预测结果

for epoch in range(num_epochs):
    # 训练模式
    model.train()
    # 前向传播
    y_pred = model(X)
    # 计算损失
    loss = criterion(y_pred, y)
    # 反向传播+优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录loss
    loss_history.append(loss.item())

    # 每500轮记录一次预测结果（减少数据量）
    if (epoch + 1) % 500 == 0:
        model.eval()
        with torch.no_grad():
            pred_history.append(model(X).numpy())
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# ===================== 最终预测 =====================
model.eval() #将模型从「训练模式（train）」切换为「评估 / 预测模式（eval）」
with torch.no_grad(): #关闭梯度计算
    y_pred_final = model(X).numpy()  # 最终预测结果

# ===================== 可视化结果 =====================
plt.figure(figsize=(10, 12))
# 子图1：Loss变化曲线
plt.subplot(2, 1, 1)
plt.plot(range(num_epochs), loss_history, color='blue', linewidth=1)
plt.xlabel('训练轮数 (Epoch)')
plt.ylabel('均方误差损失 (MSE Loss)')
plt.title('训练过程中Loss的变化')
plt.grid(True, alpha=0.3)

# 子图2：sin函数拟合效果
plt.subplot(2, 1, 2)
# 绘制原始数据点
plt.scatter(x_numpy, y_numpy, label='带噪声的原始sin数据', color='lightblue', alpha=0.6, s=10)
# 绘制真实sin曲线
plt.plot(x_numpy, np.sin(x_numpy), label='真实sin(x)曲线', color='green', linewidth=2, linestyle='--')
# 绘制模型拟合的曲线
plt.plot(x_numpy, y_pred_final, label='多层网络拟合曲线', color='red', linewidth=2)
plt.xlabel('x (0 ~ 2π)')
plt.ylabel('y = sin(x) + 噪声')
plt.title('多层神经网络拟合sin函数的效果')
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, alpha=0.3)

# 调整子图间距
plt.tight_layout()
plt.savefig("model_sin.png")  # 保存图片
plt.show()

# plt.figure(figsize=(10, 6))
# plt.scatter(x_numpy, y_numpy, label='带噪声的原始数据', color='lightblue', alpha=0.6, s=10)
# plt.plot(x_numpy, np.sin(x_numpy), label='真实sin(x)', color='green', linewidth=2, linestyle='--')
# plt.plot(x_numpy, y_pred_final, label='多层网络拟合结果', color='red', linewidth=2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('sin函数拟合最终效果')
# plt.legend()
# plt.grid(True)
# plt.show()
