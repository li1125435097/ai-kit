import numpy as np

# 创建一个一维数组，元素为1,2,3,4,5，数据类型为int32
a = np.array([1,2,3,4,5])
print(a,a.dtype)

# 创建一个一维数组，元素为1.1,2.2,3.3,4.4,5.5，数据类型为float64
a = np.array([1.1,2.2,3.3,4.4,5.5])
print(a,a.dtype)

# 创建一个3行4列的数组，元素全部为0，数据类型为float64
a = np.zeros((3,4))
print(a,a.dtype)

# 创建一个3行4列的数组，元素全部为1，数据类型为float64
a = np.ones((3,4))
print(a,a.dtype)

# 创建一个2行2列的数组，元素全部为空，数据类型为float64
a = np.empty((2,2))
print(a,a.dtype)

# 创建一个一维数组，元素为10到19，步长为3，数据类型为int32
a = np.arange(10,20,3)
print(a,a.dtype)

# 创建一个一维数组，元素为0到1，步长为0.2，数据类型为float64
a = np.arange(0,1,0.2)
print(a,a.dtype)

# 创建一个一维数组，元素为0到1，等分为5个，数据类型为float64
a = np.linspace(0,1,5)
print(a,a.dtype)

# 创建一个一维数组，元素为0到2π，等分为20个，数据类型为float64
a = np.linspace(0,2*np.pi,20)
print(a,a.dtype)

# 对数组a进行正弦运算，数据类型为float64
a = np.sin(a)
print(a,a.dtype)
