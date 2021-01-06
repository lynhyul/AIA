#keras23_3을 카피해서
#LSTM층을 두개 만들 것
#model.add(LSTM(10, input_shape=(3,1)))
#model.add(LSTM(10))


import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], 
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])   # (3,)

print(x.shape)      # (13,3)
print(y.shape)      # (13,)


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaler.fit(x)
#x = scaler.transform(x)
#x_pred = x_pred.reshape(1,-1)
#x_pred = scaler.transform(x_pred)

print(x.shape[0])   # 13
print(x.shape[1])   # 3
#x = x.reshape(13,3,1)
x = x.reshape(x.shape[0],x.shape[1],1)  # (13, 3, 1)

#코딩 하시오!! LSTM
#나는 80을 원하고 있다.

#2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(40, activation= 'relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(40, activation= 'relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(40, activation= 'relu', input_shape=(3,1), return_sequences=False))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(1))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 40)             6720
_________________________________________________________________
lstm_1 (LSTM)                (None, 3, 40)             12960
_________________________________________________________________
lstm_2 (LSTM)                (None, 40)                12960
_________________________________________________________________
dense (Dense)                (None, 30)                1230
_________________________________________________________________
dense_1 (Dense)              (None, 20)                620
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 21
=================================================================
Total params: 34,511
Trainable params: 34,511
Non-trainable params: 0
_________________________________________________________________
'''

# 컴파일, 훈련


#from tensorflow.keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x,y,epochs=400, batch_size=5)
#callbacks = early_stopping)

# 평가 및 예측

loss = model.evaluate(x,y)
print("loss, mae : " ,loss)

x_pred = x_pred.reshape(1,3,1)
result = model.predict(x_pred)
print(result)

'''
model.add(LSTM(40, activation= 'relu', input_shape=(3,1), 
return_sequences=False))
loss, mae :  [0.00102536054328084, 0.026773013174533844]
[[79.76837]]


model.add(LSTM(40, activation= 'relu', input_shape=(3,1), 
return_sequences=True))
loss, mae :  [0.007753197569400072, 0.07041938602924347]
[[80.01562]]
'''