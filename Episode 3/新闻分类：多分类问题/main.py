from keras.datasets import reuters
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


# 将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimentions=10000):
    results = np.zeros((len(sequences), dimentions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# 标签的one_hot编码
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# 数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 标签向量化
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

layers = [
    Dense(64, activation='relu', input_shape=(10000,)),
    Dense(64, activation='relu'),
    Dense(46, activation='softmax')
]
model = Sequential(layers)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=512,
                    epochs=20,
                    validation_data=[x_val, y_val])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Train loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Train and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
