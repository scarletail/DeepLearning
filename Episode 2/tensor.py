import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

'''
向量和张量维度区分：
3D向量： [1, 2, 3]
3D张量： 三维数组
'''

'''
张量的关键属性：
    1.轴的个数（阶）
    2.形状
    3.数据类型
'''

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 轴的个数
print(train_images.ndim)
# 形状
print(train_images.shape)
# 数据类型
print(train_images.dtype)

# 画出3D张量中第4个数字
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
