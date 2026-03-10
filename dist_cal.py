import numpy as np

points = np.fromfile('000000.bin', dtype=np.float32).reshape(-1, 4)
print(points.shape)
#### 特征可视化

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

skip = 10   # Skip every n points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
point_range = range(0, points.shape[0], skip) # skip points to prevent crash
x = points[point_range, 0]
y = points[point_range, 1]
z = points[point_range, 2]
ax.scatter(x,   # x
           y,   # y
           z,   # z
           c=z, # height data for color
           cmap='Blues',
           marker="o")
ax.axis('auto')  # {equal, scaled}
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('feature visulization')
ax.axis('on')          # 设置坐标轴不可见
ax.grid(True)          # 设置背景网格不可见

plt.show()

