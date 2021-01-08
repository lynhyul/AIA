import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

#2 model

model = Sequential()
model.add(LSTM(400,activation = 'relu',input_shape=(4,1)))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()

#model save

model.save("./model/save_keras35.h5") # .하나는 현재폴더를 뜻한다.
model.save(".//model//save_keras35_1.h5") # .하나는 현재폴더를 뜻한다.
model.save(".\model\save_keras35_2.h5") # .하나는 현재폴더를 뜻한다.
model.save(".\\model\\save_keras35_3.h5") # .하나는 현재폴더를 뜻한다.