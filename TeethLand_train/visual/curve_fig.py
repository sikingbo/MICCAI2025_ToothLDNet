import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x_values = np.linspace(0, 0.1, 11)  # x坐标点，0到0.1，共11个点
y_dgcnn = np.array([0.099, 0.1834, 0.3470, 0.4704, 0.5393, 0.5830, 0.6097, 0.6264, 0.6382, 0.6483, 0.6544])  # 对应的y坐标数据
y_pn2 = np.array([0.1171, 0.2144, 0.3791, 0.4896, 0.5646, 0.6085, 0.6427, 0.6583, 0.6673, 0.6742, 0.6804])  # 对应的y坐标数据
y_pra = np.array([0.0985, 0.1732, 0.3215, 0.4255, 0.4894, 0.5354, 0.5627, 0.5852, 0.5996, 0.6095, 0.6167])  # 对应的y坐标数据
y_skeleton = np.array([0., 0.0062, 0.044, 0.097, 0.1503, 0.2017, 0.2576, 0.3089, 0.3581, 0.4032, 0.4476])
y_ukpgan = np.array([0.013, 0.028, 0.0544, 0.0916, 0.131, 0.182, 0.2358, 0.2897, 0.3399, 0.3792, 0.3995])
y_mt = np.array([0.1564, 0.2778, 0.4730, 0.6052, 0.6980, 0.7541, 0.7866, 0.8062, 0.8156, 0.8212, 0.8271])
y_ours = np.array([0.1549, 0.2887, 0.5375, 0.7029, 0.7930, 0.8390, 0.8736, 0.8965, 0.9170, 0.9279, 0.9382])


# 创建Matplotlib图形对象
plt.figure(figsize=(8, 6))

# 绘制曲线并自定义颜色和线型
plt.plot(x_values, y_skeleton, label='SkeletonMerger', color=(128/255, 0/255, 128/255), linestyle='-', linewidth=3)  # chocolate	#D2691E	(210,105,30)
plt.plot(x_values, y_ukpgan, label='UKPGAN', color=(255/255, 165/255, 0/255), linestyle='-', linewidth=3)  # dark violet	#9400D3	(148,0,211)
plt.plot(x_values, y_pra, label='PRA-Net', color=(0/255, 139/255, 139/255), linestyle='-', linewidth=3)  # navy	#000080	(0,0,128)
plt.plot(x_values, y_pn2, label='PointNet++', color=(205/255, 173/255, 0/255), linestyle='-', linewidth=3)  # orange	#FFA500	(255,165,0)
plt.plot(x_values, y_dgcnn, label='DGCNN', color=(0/255, 0/255, 205/255), linestyle='-', linewidth=3)  # forest green	#228B22	(34,139,34)
plt.plot(x_values, y_mt, label='Wei et al.', color=(0/255, 100/255, 0/255), linestyle='-', linewidth=3)  # orange red	#FF4500	(255,69,0)
plt.plot(x_values, y_ours, label='Ours', color=(215/255, 0/255, 0/255), linestyle='-', linewidth=3)  # corn flower blue	#6495ED	(100,149,237)

# 自定义横纵坐标的刻度
plt.xticks(np.arange(0, 0.11, 0.02))  # x坐标刻度从0到0.1，每0.02一个刻度
plt.yticks(np.arange(0, 1.1, 0.2))   # y坐标刻度从0到1，每0.2一个刻度

# 添加图例
plt.legend(loc='lower right')

# 添加注释
# plt.annotate('Yellow line: DGCNN', xy=(0.06, 0.8), color='yellow', fontsize=12)

# 设置图形标题和坐标轴标签
# plt.title('Customized Axes and Data Plot')
plt.xlabel('Euclidean Distance', fontsize=12)
plt.ylabel('mIoU', fontsize=12)

# 显示网格
# plt.grid(True)

# 显示图形
plt.show()
