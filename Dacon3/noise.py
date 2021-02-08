import numpy as np
import PIL
from numpy import asarray
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
import string



# dirty 데이터는 train 데이터 훈련시키자!
# 50000개 
# dirty_mnist_2nd_answer.csv 는 dirty의 y값 


# test_dirty 데이터는 test 데이터!
# 5000개 
# y값을 찾는것이 목표


# img=[]
# for i in range(50000,55000):
#     filepath='../data/csv/Dacon3/test_dirty_mnist_2nd/%d.png'%i
#     image=Image.open(filepath)
#     image_data=asarray(image)
#     img.append(image_data)


# np.save('../data/csv/Dacon3/test.npy', arr=img)
# alphabets = string.ascii_lowercase
# alphabets = list(alphabets)


x = np.load('../data/csv/Dacon3/train.npy')
x_pred = np.load('../data/csv/Dacon3/test.npy') 

# print(x_pred.shape) # 5000,256,256
# print(x_pred.shape) # 50000,256,256
y = pd.read_csv('../data/csv/Dacon3/dirty_mnist_2nd_answer.csv')

sub = pd.read_csv('../data/csv/Dacon3/sample_submission.csv')

y = y.iloc[:20000,1:]
# # y = y['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
# #       'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] 

y = y.to_numpy()
x = x[:20000,:,:]

#전처리
x = x.reshape(-1,256,256,1)/255.
x_pred = x_pred.reshape(-1,256,256,1)/255.

#노이즈 제거?
threshold = 1
x[x < threshold] = 0
x_pred[x_pred <threshold] = 0

plt.figure(figsize=(20, 5))
ax = plt.subplot(2, 10, 1)
plt.imshow(x[0])


ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

print(x[0])