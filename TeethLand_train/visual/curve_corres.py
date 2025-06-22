import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x_values = np.linspace(0, 0.1, 11)  # x坐标点，0到0.1，共11个点
y_dgcnn = np.array([0.1006, 0.1725, 0.3370, 0.4638, 0.5392, 0.5864, 0.6170, 0.6322, 0.6417, 0.6474, 0.6501])  # 对应的y坐标数据
y_pn2 = np.array([0.1179, 0.1986, 0.3637, 0.4859, 0.5601, 0.6112, 0.6420, 0.6574, 0.6652, 0.6689, 0.6700])  # 对应的y坐标数据
y_pointconv = np.array([0.1187, 0.2270, 0.4518, 0.5855, 0.6419, 0.6764, 0.6970, 0.7058, 0.7160, 0.7204, 0.7226])
y_mt = np.array([0.1517, 0.2506, 0.4436, 0.5843, 0.6805, 0.7393, 0.7678, 0.7856, 0.7953, 0.7992, 0.8012])
y_pranet = np.array([0.1141, 0.2000, 0.3978, 0.5542, 0.6354, 0.6787, 0.7026, 0.7138, 0.7227, 0.7256, 0.7292])
y_ours = np.array([0.1146, 0.2008, 0.3972, 0.5702, 0.6818, 0.7479, 0.7874, 0.8169, 0.8388, 0.8589, 0.8709])


# 创建Matplotlib图形对象
plt.figure(figsize=(8, 6))

# 绘制曲线并自定义颜色和线型
plt.plot(x_values, y_pointconv, label='PointConv', color=(128/255, 0/255, 128/255), linestyle='-', linewidth=3)  # deep pink	#FF1493	(255,20,147)
plt.plot(x_values, y_pranet, label='PRA-Net', color=(0/255, 139/255, 139/255), linestyle='-', linewidth=3)  # orange red	#FF4500	(255,69,0)
plt.plot(x_values, y_pn2, label='PointNet++', color=(205/255, 173/255, 0/255), linestyle='-', linewidth=3)  # orange	#FFA500	(255,165,0)
plt.plot(x_values, y_dgcnn, label='DGCNN', color=(0/255, 0/255, 205/255), linestyle='-', linewidth=3)  # forest green	#228B22	(34,139,34)
plt.plot(x_values, y_mt, label='Wei et al.', color=(0/255, 100/255, 0/255), linestyle='-', linewidth=3)  # light sea green	#20B2AA	(32,178,170)
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
