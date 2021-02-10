
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


x = np.load('../data/csv/Dacon3/train2.npy')
x_pred = np.load('../data/csv/Dacon3/test2.npy') 

# print(x_pred.shape) # 5000,256,256
# print(x_pred.shape) # 50000,256,256
y = pd.read_csv('../data/csv/Dacon3/dirty_mnist_2nd_answer.csv')

sub = pd.read_csv('../data/csv/Dacon3/sample_submission.csv')

y = y.iloc[:20000,1:]
# # y = y['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
# #       'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] 

y = y.to_numpy()


#전처리
x = x.reshape(-1,128,128,1)/255.
x_pred = x_pred.reshape(-1,128,128,1)/255.




x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8)



# 이미지 증폭
idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),   # 0.1 => acc 하락
    height_shift_range=(-1,1),  # 0.1 => acc 하락
    fill_mode = 'nearest')
    # rotation_range=40, 
    # shear_range=0.2,    # 현상유지
    # zoom_range=0.2) 
    # horizontal_flip=True)

idg2 = ImageDataGenerator()

'''
- rotation_range: 이미지 회전 범위 (degrees)
- width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 
                                (원본 가로, 세로 길이에 대한 비율 값)
- rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 
            모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
- shear_range: 임의 전단 변환 (shearing transformation) 범위
- zoom_range: 임의 확대/축소 범위
- horizontal_flip`: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 
    원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
- fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''

train_generator = idg.flow(x_train,y_train,batch_size=32, seed = 2000)
# seed => random_state
valid_generator = idg2.flow(x_test,y_test)
test_generator = idg2.flow(x_pred,shuffle=False)
model = Sequential()

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size =(3,3), activation='sigmoid', padding = 'same', 
                                            input_shape=(128,128,1)))
model.add(BatchNormalization())                                  
model.add(Conv2D(filters = 32, kernel_size =(3,3), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size =(5,5), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size =(5,5), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

                               
model.add(Conv2D(filters = 64, kernel_size =(3,3), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size =(5,5), padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation= 'relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation= 'relu'))
model.add(BatchNormalization())
model.add(Dense(26, activation='sigmoid'))
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
mc = ModelCheckpoint('../data/modelcheckpoint/Daconproject1.h5',save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience= 160)
lr = ReduceLROnPlateau(patience= 80, factor=0.5)
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.002,epsilon=None),
                    metrics=['acc'])
learning_history = model.fit_generator(train_generator,epochs=1000, 
    validation_data=valid_generator, callbacks=[early_stopping,lr])

result = model.predict_generator(test_generator,verbose=True)
result[result < 0.5] =0
result[result > 0.5] =1
sub.iloc[:,1:] = result
sub.to_csv('../data/csv/Dacon3/Dacon8.csv',index=False)
