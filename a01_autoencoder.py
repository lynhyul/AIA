import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data() # y는 사용하지 않지만 자리는 _로 명시한다

x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,784).astype('float32')/255.

# print(x_train[0])

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64,activation='relu') (input_img)
decoded = Dense(784,activation='sigmoid') (encoded)

autoencoder = Model(input_img, decoded)

autoencoder.summary()
'''
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
dense (Dense)                (None, 64)                50240
_________________________________________________________________
dense_1 (Dense)              (None, 784)               50960
=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
_________________________________________________________________
'''
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=['acc'])
# autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train,x_train, epochs=30, batch_size= 256, validation_split=0.2)
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n= 10
for i in range(n) :
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
