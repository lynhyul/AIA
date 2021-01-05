import numpy as np

# 1.데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], 
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])   # (3,)

print(x.shape)      # (13,3)
print(y.shape)      # (13,)

x = x.reshape(13,3,1)

# 2.모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(10, activation='relu', input_shape = (3,1)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x,y,epochs=300, batch_size=8)

#4 예측, 평가

loss = model.evaluate(x,y)
print(loss)

x_pred = x_pred.reshape(1,3,1)
result = model.predict(x_pred)
print(result)