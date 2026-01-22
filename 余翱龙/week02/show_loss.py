import matplotlib.pyplot as plt
import numpy as np

# 数据
data_sets = [
    ['2.4150', '2.2078', '1.8979', '1.5226', '1.1947', '0.9637', '0.8124', '0.7109', '0.6377', '0.5817'],
    ['2.4148', '2.2205', '1.9147', '1.5453', '1.2173', '0.9802', '0.8240', '0.7168', '0.6434', '0.5861'],
    ['2.4396', '2.3735', '2.3025', '2.1612', '1.8688', '1.4927', '1.1439', '0.8760', '0.7052', '0.5942'],
    ['2.4425', '2.3791', '2.3087', '2.1833', '1.9190', '1.5072', '1.1317', '0.8617', '0.6918', '0.5845'],
    ['2.3300', '1.9953', '1.5539', '1.1627', '0.9105', '0.7561', '0.6549', '0.5839', '0.5292', '0.4865'],
    ['2.3279', '1.9739', '1.5196', '1.1314', '0.8909', '0.7408', '0.6479', '0.5758', '0.5238', '0.4825']
]

# 将字符串转换为浮点数
data_float = []
for data_set in data_sets:
    data_float.append([float(x) for x in data_set])

# x轴数据（假设是0-9）
x = list(range(10))

# 创建图形
plt.figure(figsize=(12, 8))

# 颜色和线型
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
line_styles = ['-', '--', '-.', ':', '-', '--']
labels = ['1layer-128', '1layer-256', '2layer-128', '2layer-256', '2layer-128-res', '2layer-256-res']

# 绘制每条折线
for i, data in enumerate(data_float):
    plt.plot(x, data,
             color=colors[i % len(colors)],
             linestyle=line_styles[i % len(line_styles)],
             marker='o',
             markersize=4,
             linewidth=2,
             label=labels[i])

# 设置图表属性
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('different hid_layer & hid_dim', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(x)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 也可以保存图表
# plt.savefig('different hid_layer & hid_dim', dpi=300, bbox_inches='tight')