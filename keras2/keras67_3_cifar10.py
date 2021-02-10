# cifar10 => flow
# ImageDataGenerator


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD,Adadelta
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import cifar10


(x_train,y_train),(x_valid,y_valid) = cifar10.load_data()

print(x_train.shape)    #50000,32,32,3
print(y_train.shape)    #50000,1
print(x_valid.shape)     #10000,32,32,3

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_valid = y_valid.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_valid = one.transform(y_valid).toarray()


idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),   # 0.1 => acc 하락
    height_shift_range=(-1,1),  # 0.1 => acc 하락
    # rotation_range=40, acc 하락 
    # shear_range=0.2)    # 현상유지
    # zoom_range=0.2,
    horizontal_flip=True)

idg2 = ImageDataGenerator()

train_generator = idg.flow(x_train,y_train,batch_size=32, seed = 2048)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size =(3,3), activation='relu', padding = 'same', 
                                        input_shape=(32,32,3)))
model.add(Dropout(0.5))                                
model.add(Conv2D(filters = 32, kernel_size =(3,3), padding = 'same', activation='relu'))
model.add(Dropout(0.5))  
model.add(Conv2D(filters = 32, kernel_size =(5,5), padding = 'same', activation='relu'))
model.add(Dropout(0.5))  
model.add(Conv2D(filters = 32, kernel_size =(5,5), padding = 'same', activation='relu'))
model.add(Dropout(0.5))  
model.add(MaxPooling2D(2,2))

                            
model.add(Conv2D(filters = 64, kernel_size =(3,3), padding = 'same', activation='relu'))
model.add(Dropout(0.5))  
model.add(Conv2D(filters = 64, kernel_size =(5,5), padding = 'same', activation='relu'))
model.add(Dropout(0.5))  
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.5))  
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.5))  
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(patience= 50)
lr = ReduceLROnPlateau(patience= 25, factor=0.5)

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=0.002,epsilon=None),
                metrics=['acc'])
history = model.fit_generator(train_generator,epochs=200, 
    validation_data=valid_generator, callbacks=[early_stopping,lr])



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss'] 
val_loss = history.history['val_loss']


print('acc : ', acc[-1])
print('val_acc : ', val_acc[:-1])

'''
acc :  0.09994000196456909
'''