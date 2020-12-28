# 네이밍 룰
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.arange(1,11)
y_train = np.arange(2,21,2)
x_test  = np.arange(101,111)
y_test  = np.arange(111,121)

x_predict = np.arange(111,114)


#2. 모델구성
model = Sequential()
model.add(Dense(10000, input_dim=1, activation='sigmoid' ))
model.add(Dense(5000, input_dim=1, activation='relu' ))
i =0
for i in range(1,20) :
    model.add(Dense(40-i*2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs =200, batch_size =10)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=10)
print("loss : ", loss)

result = model.predict([x_predict])
print("result : ", result)

