import warnings
warnings.filterwarnings('ignore')

## Basic Import ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

## Tensorflow ## 
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array # Image Related


train=pd.read_csv('../data/csv/practice/train.csv').set_index('id')


X_image=train.iloc[:,2:].values.reshape(-1,28,28,1)
X_letter=train.letter
Y=to_categorical(train.digit)


# 이미지 생성기의 선언
datagen = ImageDataGenerator(
                                 width_shift_range=5,
                                 height_shift_range=5,
                                 rotation_range=10,
                                 zoom_range=0.05)  


# flow형태의 정의
flow1=datagen.flow(X_image,X_letter,batch_size=32,seed=2020) 
flow2=datagen.flow(X_image,Y,batch_size=32,seed=2020)

# flow를 통한 이미지, 글자, 라벨의 생성
X_image_gen1,X_letter_gen=flow1.next()
X_image_gen2,Y_gen=flow2.next()

## 생성된 데이터의 형태 확인
print("X_image_gen1.shape={}".format(X_image_gen1.shape))   # 32, 28,28,1
print("letter_gen.shape={}".format(X_letter_gen.shape))
print("X_image_gen2.shape={}".format(X_image_gen2.shape))
print("Y_gen.shape={}".format(Y_gen.shape))

temp = pd.DataFrame(train)
x = temp.iloc[:,3:]/255
x = x.to_numpy()

X_image_gen1 = X_image_gen1.reshape(32,28*28)

# x = np.append(x,X_image_gen1)

print(x.shape)