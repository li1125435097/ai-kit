import numpy as np

# 创建一个从20到50，步长为10的数组
a = np.arange(20,51,10)
print(a)
# 创建一个从1到4的数组
b = np.arange(4)+1
print(b)
# 计算a和b的差
print(a-b)
# 计算b的平方
print(b**2)

# 创建一个从0到180，步长为30的数组
c = np.arange(0,181,30)
# 将数组中的元素转换为弧度
c = c*np.pi/180
print(c)
# 计算c的正弦值
print(np.sin(c))
# 判断a中的元素是否大于30
print(a>30)
# 计算a和b的乘积
print(a*b)
# 计算a和b的矩阵乘积
print(a@b)
# 计算a中所有元素的和
print(a.sum())
# 计算a重新排列为2行2列的矩阵后，每一列的和
print(a.reshape(2,2).sum(axis=0))
# 计算a重新排列为2行2列的矩阵后，每一行的和
print(a.reshape(2,2).sum(axis=1))

# 将a的形状改变为2行2列
d = a.view().reshape(2,2)
# 打印d和a的值
print(d,a)
# 将d的第0行第0个元素赋值为10
d[0,0] = 10
# 打印d和a的值
print(d,a)

e = a.copy().reshape(2,2)
# 将e的第0行第0个元素赋值为10
a[0] = 20
# 打印e和a的值
print(e,a)
