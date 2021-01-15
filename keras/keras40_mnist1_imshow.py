# 인공지능계의 hello world라 불리는 mnist!!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000,28,28) (60000,)    => 60000,28,28,1
print(x_test.shape, y_test.shape)       # (10000,28,28) (10000,)    => 10000,28,28,1

print(x_train[0])   #    
print(y_train[0])   # 5

print(x_train[0].shape)     #(28, 28)

plt.imshow(x_train[0], 'gray')
plt.show()

# 인공지능계의 hello world라 불리는 mnist!!!

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000,28,28) (60000,)    => 60000,28,28,1
print(x_test.shape, y_test.shape)       # (10000,28,28) (10000,)    => 10000,28,28,1

print(x_train[0])   #    
print(y_train[0])   # 5

print(x_train[0].shape)     #(28, 28)

plt.imshow(x_train[0], 'gray')
plt.show()

