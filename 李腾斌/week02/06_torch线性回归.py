import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 数据
# =========================
X_numpy = np.random.rand(200, 1) * 10
y_numpy = 2 * np.sin(X_numpy) + 1 + np.random.randn(200, 1) * 0.1  # 增加噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

# =========================
# 2. 可配置 MLP
# =========================
class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =========================
# 3. 超参数组合搜索
# =========================
layer_options = [
    [64, 64],          # 2层，每层64
    [128, 128],        # 2层，每层128
    [128, 128, 64],    # 3层
    [128, 128, 128],   # 3层
    [128, 128, 64, 32] # 4层
]

best_loss = float('inf')
best_model = None
best_config = None

num_epochs = 3000
lr = 0.01

for hidden_dims in layer_options:
    model = ConfigurableMLP(input_dim=1, hidden_dims=hidden_dims, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"Hidden dims: {hidden_dims}, Final Loss: {final_loss:.6f}")

    if final_loss < best_loss:
        best_loss = final_loss
        best_model = model
        best_config = hidden_dims

print(f"\n最佳模型 Hidden dims: {best_config}, Loss: {best_loss:.6f}")

# =========================
# 4. 用最佳模型预测
# =========================
best_model.eval()
with torch.no_grad():
    y_pred_best = best_model(X).numpy()

sorted_idx = np.argsort(X_numpy[:, 0])
X_sorted = X_numpy[sorted_idx]
y_sorted = y_numpy[sorted_idx]
y_pred_sorted = y_pred_best[sorted_idx]

plt.figure(figsize=(10,6))
plt.scatter(X_numpy, y_numpy, label='Raw data', alpha=0.6)
plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Best MLP Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
