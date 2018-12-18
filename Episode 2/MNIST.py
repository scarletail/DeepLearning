from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_labels))
print(test_labels)

# 构建网络
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

'''
编译神经网络需要选择三个参数：
    1.损失函数（loss function），网络如何衡量在训练数据上的性能，即网络如何朝着正确的方向前进
    2.优化器（optimizer），基于训练数据和损失函数来更新网络的机制
    3.在训练和测试过程中需要的指标（metric）
'''

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 数据预处理

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 准备标签

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练模型

model.fit(train_images, train_labels, epochs=5, batch_size=128)

'''
训练过程中显示了两个数字：
    网络在训练数据上的损失loss
    训练数据上的精度acc
'''

# 检查模型在测试集上的性能

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)

'''
模型在新数据上的性能往往比在训练集上要差，这种现象称之为过拟合
'''