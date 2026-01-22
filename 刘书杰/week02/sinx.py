import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn

X_numpy = np.linspace(-4*np.pi, 4*np.pi, 100).reshape(-1, 1)

y_numpy = np.sin(X_numpy)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 1
hidden_size = 128
output_size = 1
model = Net(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):

    outputs = model(X)

    # 计算损失
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")

# 6. 绘制结果
with torch.no_grad():
    predicted = model(X).detach().numpy()

plt.figure(figsize=(10,6))
plt.plot(X_numpy, y_numpy,label='True sin(x)',color='blue',linewidth=2)
plt.plot(X_numpy,predicted,label='Fitted curve',color='red',linestyle='--',linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sin(x)')

plt.legend()
plt.grid(True)
plt.show()
