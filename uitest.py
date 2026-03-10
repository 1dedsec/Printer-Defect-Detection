import numpy as np
import matplotlib.pyplot as plt

# 提供的数据
X = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 2000, 5000])
Y = np.array([63.8, 91.6, 119.4, 146.9, 174.1, 207.4, 231.1, 256.4, 282, 313.3, 335.7, 391.7, 454.8, 503.6, 564, 1084.6, 2820])

# 使用polyfit函数进行一次多项式拟合
coefficients = np.polyfit(X, Y, 1)
poly_fit = np.poly1d(coefficients)

# 生成拟合曲线上的点
fit_line = poly_fit(X)

# 绘制原始数据和拟合曲线
plt.scatter(X, Y, label='Original Data')
plt.plot(X, fit_line, color='red', label='Linear Fit')

# 添加标签和图例
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Fit')
plt.legend()

# 显示图形
plt.show()
