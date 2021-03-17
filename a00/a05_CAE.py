# 4번 카피해서 복붙
# CNN 딥하게 구성
# 2개의 모델을 만드는데 하나는 원칙적 오토 인코더
# 다른 하나는 랜덤하게 만들고 싶은대로 히든을 구성
# 2개의 성능 비교



import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data() # y는 사용하지 않지만 자리는 _로 명시한다

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_train2 = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

# print(x_train[0])

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten, GlobalAveragePooling2D,MaxPooling2D, BatchNormalization,Activation

def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size,kernel_size=(3,3), input_shape=(28,28,1),
                    activation= 'relu', padding='valid'))
    model.add(Conv2D(filters=64,kernel_size=(2,2),padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256,kernel_size=(2,2),padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(1,1))

    model.add(Flatten())
    model.add(Dense(126))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(units=784, activation='sigmoid'))
    return model


model = autoencoder(hidden_layer_size=154)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['acc'])
model.fit(x_train, x_train2, epochs= 10, batch_size= 256)

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) =\
        plt.subplots(2, 5, figsize=(20,7))

random_imges = random.sample(range(output.shape[0]), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_imges[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(x_test[random_imges[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])



plt.tight_layout()
plt.show()