from keras import models
from keras import layers
from keras import optimizers

# 序列式
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))

# 函数式
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)
model1 = models.Model(inputs=input_tensor, outputs=output_tensor)

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['accuracy'])
'''
model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
'''