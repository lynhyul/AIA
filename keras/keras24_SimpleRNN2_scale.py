#keras23_LSTM3_scale.py를 카피


import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], 
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape)      # (13,3)
print(y.shape)      # (13,)

x = x.reshape(13,3,1)

#코딩 하시오!! LSTM
#나는 80을 원하고 있다.

#2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

model = Sequential()

model.add(SimpleRNN(10, activation= 'relu', input_shape=(3,1)))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(1))

model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 10)                120
_________________________________________________________________
dense (Dense)                (None, 50)                550
_________________________________________________________________
dense_1 (Dense)              (None, 20)                1020
_________________________________________________________________

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 5)                 35
_________________________________________________________________
dense (Dense)                (None, 50)                300
_________________________________________________________________
dense_1 (Dense)              (None, 20)                1020
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 21
=================================================================

'''



# 컴파일, 훈련


#from tensorflow.keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x,y,epochs=200, batch_size=1)
#callbacks = early_stopping)

# 평가 및 예측

loss = model.evaluate(x,y)
print("loss, mae : " ,loss)

x_pred = x_pred.reshape(1,3,1)
result = model.predict(x_pred)
print(result)

'''
loss, mae :  [0.0006884553586132824, 0.020751018077135086]
[[80.10714]]
'''