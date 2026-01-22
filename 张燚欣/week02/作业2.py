import torch
import torch .nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成sin数据
x = np.linspace(-3*np.pi,3*np.pi,200).reshape(-1,1)
y = np.sin(x)+0.1*np.random.randn(200,1)

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# 简单多层网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self,x):
        return self.net(x)

# 训练
model = SimpleNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

losses = []
for epoch in range(1000):
    y_pred = model(x_tensor)
    loss = loss_fn(y_pred,y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

# 预测和画图
with torch.no_grad():
    y_fit = model(x_tensor).numpy()

plt.figure(figsize=(12,4))

# 训练损失
plt.subplot(1,3,1)
plt.plot(losses)
plt.title('训练Loss')

# 拟合效果
plt.subplot(1, 3, 2)
plt.scatter(x, y, s=10, alpha=0.5, label = '数据')
plt.plot(x, np.sin(x), 'g-', label = '真实sin')
plt.plot(x, y_fit, 'r-', label = '拟合')
plt.legend()
plt.title('拟合效果')

# 误差
plt.subplot(1,3,3)
error = np.abs(y_fit - np.sin(x))
plt.plot(x,error,'purple')
plt.title('拟合误差')

plt.tight_layout()
plt.show()

print(f"最终Loss:{losses[-1]:.6f}")