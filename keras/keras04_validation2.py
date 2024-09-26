from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import arange,array
#np.array()
#array()

#1. 데이터
x_train = arange(1,11)
y_train = arange(1,11)
x_test = arange(11,16)
y_test = arange(11,16)
x_pred = array([16,17,18])


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim =1, activation='relu'))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics = 'mae')
model.fit(x_train, y_train, epochs = 100, batch_size=1, validation_split=0.2)   #x_train , y_train을 validation_split =0.2로 20%를 나누어쓰겠다.


#4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print("results= ",results)

y_pred = model.predict(x_pred)          #x_pred를 넣어서 얻은 결과값은 y_pred
print("y_predict :", y_pred)

