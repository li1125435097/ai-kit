import matplotlib.pyplot as plt  
x = [1, 2, 3, 4, 5]  
y = [3, 5, 9, 6, 8]  
plt.plot(x, y, marker='o', linestyle='--', color='#2E86C1')  
plt.title("2025 年销售趋势图")  
plt.xlabel("季度")  
plt.ylabel("销售额（万元）")  
plt.grid(True)  
plt.show()  

import numpy as np  
x = np.random.rand(50) * 10  
y = x * 2 + np.random.normal(0, 2, 50)  
plt.scatter(x, y, s=80, c=y, cmap='viridis', edgecolors='black')  
plt.colorbar(label='数值强度')  
plt.title("变量相关性分析")  
plt.show()  

categories = ['北京', '上海', '广州', '深圳']  
values = [120, 95, 80, 105]  
plt.bar(categories, values, color=['#E74C3C', '#3498DB', '#2ECC71', '#F1C40F'])  
plt.title("2025 年城市 GDP 对比")  
plt.xticks(rotation=45)  
plt.show()  

labels = ['电子产品', '服装', '食品', '其他']  
sizes = [45, 30, 15, 10]  
explode = (0.1, 0, 0, 0)  # 突出显示最大类别  
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)  
plt.title("电商平台品类占比")  
plt.show()  

data = np.random.randn(1000)  
plt.hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')  
plt.title("数据分布直方图")  
plt.xlabel("数值区间")  
plt.ylabel("频率密度")  
plt.show()  


x = np.arange(5)  
y = [10, 12, 8, 14, 9]  
error = [1.2, 0.8, 1.5, 0.6, 1.0]  
plt.errorbar(x, y, yerr=error, fmt='o', capsize=5, color='purple')  
plt.title("实验测量误差展示")  
plt.show()  

x = [1, 2, 3, 4, 5]  
y1 = [1, 3, 4, 7, 6]  
y2 = [2, 4, 5, 8, 7]  
plt.stackplot(x, y1, y2, labels=['A 产品', 'B 产品'], alpha=0.5)  
plt.legend(loc='upper left')  
plt.title("产品累积销售额")  
plt.show()  


data = [np.random.normal(0, std, 100) for std in range(1, 4)]  
plt.boxplot(data, patch_artist=True, notch=True)  
plt.xticks([1, 2, 3], ['组1', '组2', '组3'])  
plt.title("多组数据分布对比")  
plt.show()  


import numpy as np  
matrix = np.random.rand(5, 5)  
plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')  
plt.colorbar()  
plt.title("相关性热力图")  
plt.xticks(range(5), ['A', 'B', 'C', 'D', 'E'])  
plt.yticks(range(5), ['X', 'Y', 'Z', 'W', 'V'])  
plt.show()  


from mpl_toolkits.mplot3d import Axes3D  
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  
z = np.linspace(0, 15, 100)  
x = np.sin(z)  
y = np.cos(z)  
ax.plot(x, y, z, color='#E67E22')  
ax.set_title("螺旋三维曲线")  
plt.show()  


theta = np.linspace(0, 2*np.pi, 8)  
r = [1, 2, 1.5, 3, 2.5, 1, 0.5, 2]  
ax = plt.subplot(111, polar=True)  
ax.plot(theta, r, marker='D')  
ax.set_title("极坐标雷达图")  
plt.show()  


x, y = np.meshgrid(np.arange(-2, 2, 0.5), np.arange(-2, 2, 0.5))  
u = np.sin(x)  
v = np.cos(y)  
plt.quiver(x, y, u, v, scale=20, color='red')  
plt.title("矢量场示意图")  
plt.show()  




