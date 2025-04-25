import numpy as np


# 创建一个包含15个元素的数组，并将其重塑为5行3列的二维数组
a = np.arange(15).reshape(5,3)
print(dir(a))
print(a)

print('输出数组的形状：',a.shape)
print('输出数组的维度：',a.ndim)
print('输出数组的元素类型：',a.dtype)
print('输出数组的元素个数：',a.size)
print('输出数组的第2行：',a[1])
print('输出数组的第2行第3列元素：',a[1,2])
print('输出数组的第2列：',a[:,1])
print('输出类型：',type(a))
print(a.itemsize) # 输出数组中每个元素的字节大小
print(a.size) # 输出数组中元素的总数
print(a.dtype.name) 