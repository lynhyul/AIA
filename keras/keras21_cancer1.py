#유방암 예측 모델을 만들어보자
#sigmoid에 대해서 조사해보자
#전처리는 알아서 할 것
#실습1. acc를 0.985이상으로 올려볼것
#실습2. predict값 출력해볼것


import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(datasets.DESCR)
print(datasets.feature_names)
print(x.shape)      # 569,30
print(y.shape)      # 569,
print(x[:5])
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 110)




#전처리는 알아서 할 것

#2. modeling
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, activation= 'relu', input_shape = (30,) ))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

#compile, fit
#from tensorflow.keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='loss', patience= 15, mode = 'auto')
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics =['acc'])
model.fit(x_train,y_train, epochs = 300, batch_size=10, validation_split = 0.3) 
                                #callbacks = early_stopping)

loss = model.evaluate(x_test,y_test, batch_size=1)
print("loss : ",loss)

y_predict = model.predict(x[-5:-1])
print(y_predict)
print(y[-5:-1])