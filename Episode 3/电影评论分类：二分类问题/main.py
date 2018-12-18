from keras.datasets import imdb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics, losses, optimizers
import matplotlib.pyplot as plt


# 将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimentions=10000):
    results = np.zeros((len(sequences), dimentions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


'''
Note:
    np.zeros()函数输入形状时要加括号，否则会报错：
    TypeError: data type not understood
'''

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# word_index是一个将单词映射为整数索引的字典
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decode_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
)

# 数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 定义模型
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 编译模型
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 配置优化器
model.compile(optimizer=optimizers.RMSprop(
    lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# 使用自定义的损失和指标
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
# 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
# 训练模型
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Train loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
