#kras23_LSTM3_scale을 DNN으로 코딩
#결과치 비교

#DNN으로 23번 파일보다 loss를 좋게 만들것.


import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], 
            [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])   #(3,)

print(x.shape)      # (13,3)
print(y.shape)      # (13,)


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaler.fit(x)
#x = scaler.transform(x)
#x_pred = x_pred.reshape(1,-1)
#x_pred = scaler.transform(x_pred)


#코딩 하시오!! LSTM
#나는 80을 원하고 있다.

#2. 모델 구성



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()

model.add(Dense(10, activation= 'relu', input_shape=(3,)))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(1))

# 컴파일, 훈련


#from tensorflow.keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='loss', patience=28, mode='auto')
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x,y,epochs=200, batch_size=1) 
#callbacks = early_stopping)

# 평가 및 예측

loss = model.evaluate(x,y)
print("loss, mae : " ,loss)

x_pred= x_pred.reshape(1,-1)
result = model.predict(x_pred)
print(result)


'''
loss, mae :  [0.0007570709567517042, 0.017534183338284492]
[[80.05375]]
'''