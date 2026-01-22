# 配置中文字体（放在最开头）
import matplotlib.pyplot as plt

# 简单配置（适合大多数情况）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 生成sin函数数据
np.random.seed(42)
torch.manual_seed(42)

# 生成数据点
X_numpy = np.random.uniform(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)  # 在[-2π, 2π]范围内生成点
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数加上一些噪声

# 划分训练集和测试集
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_numpy, y_numpy, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train = torch.from_numpy(X_train_np).float()
y_train = torch.from_numpy(y_train_np).float()
X_test = torch.from_numpy(X_test_np).float()
y_test = torch.from_numpy(y_test_np).float()

print("数据生成完成。")
print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")
print("---" * 10)

# 2. 定义不同复杂度的神经网络模型
class SimpleSinModel(nn.Module):
    """简单模型：1个隐藏层"""

    def __init__(self, hidden_size=10):
        super(SimpleSinModel, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MediumSinModel(nn.Module):
    """中等模型：2个隐藏层"""

    def __init__(self, hidden1=20, hidden2=10):
        super(MediumSinModel, self).__init__()
        self.fc1 = nn.Linear(1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class ComplexSinModel(nn.Module):
    """复杂模型：3个隐藏层，使用Dropout"""

    def __init__(self, hidden1=64, hidden2=32, hidden3=16):
        super(ComplexSinModel, self).__init__()
        self.fc1 = nn.Linear(1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x


class SinModelWithTanh(nn.Module):
    """使用Tanh激活函数（更适合周期函数）"""

    def __init__(self, hidden1=40, hidden2=20):
        super(SinModelWithTanh, self).__init__()
        self.fc1 = nn.Linear(1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x


# 3. 训练函数
def train_model(model, model_name, X_train, y_train, X_test, y_test, num_epochs=3000, lr=0.001):
    print(f"\n{'=' * 60}")
    print(f"训练模型: {model_name}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"{'=' * 60}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算测试集损失
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test)

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'训练损失: {loss.item():.6f}, '
                  f'测试损失: {test_loss.item():.6f}')

    return model, train_losses, test_losses


# 4. 训练所有模型
models = {
    '简单模型(1层,10节点)': SimpleSinModel(hidden_size=10),
    '中等模型(2层,20-10节点)': MediumSinModel(hidden1=20, hidden2=10),
    '复杂模型(3层,64-32-16节点)': ComplexSinModel(hidden1=64, hidden2=32, hidden3=16),
    'Tanh模型(2层,40-20节点)': SinModelWithTanh(hidden1=40, hidden2=20)
}

results = {}

for name, model in models.items():
    trained_model, train_losses, test_losses = train_model(
        model, name, X_train, y_train, X_test, y_test,
        num_epochs=3000, lr=0.001
    )
    results[name] = {
        'model': trained_model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    }

# 5. 可视化训练过程
plt.figure(figsize=(15, 10))

# 绘制损失曲线
colors = ['blue', 'green', 'red', 'purple']
for idx, (name, color) in enumerate(zip(results.keys(), colors)):
    plt.subplot(2, 2, 1)
    plt.plot(results[name]['train_losses'], label=name, color=color, alpha=0.8)

    plt.subplot(2, 2, 2)
    plt.plot(results[name]['test_losses'], label=name, color=color, alpha=0.8)

plt.subplot(2, 2, 1)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('训练损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数坐标更好地观察

plt.subplot(2, 2, 2)
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('测试损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# 6. 绘制拟合结果对比
# 生成用于可视化的密集点
X_vis_np = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
X_vis = torch.from_numpy(X_vis_np).float()

plt.subplot(2, 1, 2)
# 绘制真实sin函数
y_true = np.sin(X_vis_np)
plt.plot(X_vis_np, y_true, 'k-', linewidth=3, label='True sin(x)', alpha=0.7)

# 绘制各模型的预测结果
for idx, (name, color) in enumerate(zip(results.keys(), colors)):
    model = results[name]['model']
    model.eval()
    with torch.no_grad():
        y_pred_vis = model(X_vis).numpy()
    plt.plot(X_vis_np, y_pred_vis, label=name, color=color, alpha=0.8, linewidth=1.5)

plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('不同模型的拟合效果对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 7. 打印最终结果表格
print("\n" + "=" * 80)
print("模型性能对比总结:")
print("=" * 80)
print(f"{'模型名称':<25} {'参数数量':<15} {'训练损失':<15} {'测试损失':<15} {'过拟合程度':<15}")
print("-" * 80)

for name in results.keys():
    model = results[name]['model']
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_train = results[name]['final_train_loss']
    final_test = results[name]['final_test_loss']
    overfit_ratio = final_test / final_train if final_train > 0 else float('inf')

    print(f"{name:<25} {params:<15} {final_train:<15.6f} {final_test:<15.6f} {overfit_ratio:<15.3f}")

print("=" * 80)

# 8. 绘制局部放大图（查看细节）
plt.figure(figsize=(12, 8))
X_detail_np = np.linspace(0, np.pi, 200).reshape(-1, 1)
X_detail = torch.from_numpy(X_detail_np).float()

# 绘制真实值
y_true_detail = np.sin(X_detail_np)
plt.plot(X_detail_np, y_true_detail, 'k-', linewidth=4, label='True sin(x)', alpha=0.7)

# 绘制各模型的预测
for idx, (name, color) in enumerate(zip(results.keys(), colors)):
    model = results[name]['model']
    model.eval()
    with torch.no_grad():
        y_pred_detail = model(X_detail).numpy()
    plt.plot(X_detail_np, y_pred_detail, '--', label=name, color=color, alpha=0.8, linewidth=2)

# 绘制训练数据点
plt.scatter(X_train_np[:100], y_train_np[:100], alpha=0.3, color='gray', s=20, label='Training points')

plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('拟合效果细节展示 (x ∈ [0, π])')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 9. 误差分布分析
plt.figure(figsize=(12, 8))
for idx, (name, color) in enumerate(zip(results.keys(), colors)):
    model = results[name]['model']
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test).numpy()

    errors = np.abs(y_test_pred - y_test_np)

    plt.subplot(2, 2, idx + 1)
    plt.hist(errors, bins=30, alpha=0.6, color=color, edgecolor='black')
    plt.xlabel('绝对误差')
    plt.ylabel('频率')
    plt.title(f'{name}\n平均误差: {np.mean(errors):.4f}')
    plt.grid(True, alpha=0.3)

plt.suptitle('各模型在测试集上的误差分布', fontsize=14)
plt.tight_layout()

plt.show()

# 10. 添加交互式可视化（可选）
print("\n分析总结:")
print("1. 模型复杂度 vs 拟合能力: 复杂模型通常能更好地拟合非线性函数")
print("2. 激活函数选择: Tanh可能更适合周期函数，但ReLU通常训练更快")
print("3. 过拟合风险: 检查训练损失和测试损失的差异")
print("4. 实际应用建议: 根据具体需求平衡模型复杂度和计算成本")