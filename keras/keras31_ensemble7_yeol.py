#행이 다른 앙상블 모델에 대해서 공부해보자!

import numpy as np

x1 = np.array([[1,2], [2,3], [3,4], [4,5], [5,6],
            [6,7], [7,8], [8,9], 
            [9,10], [10,11],
            [20,30], [30,40], [40,50]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70],
            [60,70,80], [70,80,90], [80,90,100], 
            [90,100,110], [100,110,120],
            [2,3,4], [3,4,5], [4,5,6]])  
y1 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70],
            [60,70,80], [70,80,90], [80,90,100], 
            [90,100,110], [100,110,120],
            [2,3,4], [3,4,5], [4,5,6]])
y2 = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_pred = np.array([55,65])   # (3,)
x2_pred = np.array([65,75,85])   # (3,)


print(x1.shape)      # (13,2)
print(x2.shape)      # (13,3)
print(y1.shape)      # (13,)
print(y2.shape)      # (13,)


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x1)
# scaler.fit(x2)
# x1 = scaler.transform(x1)
# x2 = scaler.transform(x2)
# x1_pred = x1_pred.reshape(1,-1)
# x2_pred = x2_pred.reshape(1,-1)
# x2_pred = scaler.transform(x1_pred)
# x2_pred = scaler.transform(x2_pred)


# from sklearn.model_selection import train_test_split
# x1_train, x1_test, y1_train,y1_test = train_test_split(x1,y1,test_size=0.2,
#                                     random_state = 101)
# x2_train, x2_test, y2_train,y2_test = train_test_split(x2,y2,test_size=0.2,
#                                     random_state = 101)

x1 = x1.reshape(13,2,1)
x2 = x2.reshape(13,3,1)


#코딩 하시오!! LSTM
#나는 80을 원하고 있다.

#2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM , Input

# model.add(LSTM(40, activation= 'relu', input_shape=(3,1)))
# model.add(Dense(50, activation= 'relu'))
# model.add(Dense(30, activation= 'relu'))
# model.add(Dense(20, activation= 'relu'))
# model.add(Dense(1))

input1 = Input(shape = (2,1))
lstm1 = LSTM(40, activation= 'relu') (input1)
dense1 = Dense(40, activation= 'relu') (lstm1)
dense1 = Dense(30, activation= 'relu') (dense1)
dense1 = Dense(20, activation= 'relu') (dense1)
dense1 = Dense(20, activation= 'relu') (dense1)
dense1 = Dense(20, activation= 'relu') (dense1)

input2 = Input(shape = (3,1))
lstm2 = LSTM(40, activation= 'relu') (input2)
dense2 = Dense(20, activation= 'relu') (lstm2)
dense2 = Dense(30, activation= 'relu') (dense2)
dense2 = Dense(40, activation= 'relu') (dense2)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1,dense2])
middle = Dense(10, activation= 'relu') (merge1)


output1 = Dense(100, activation= 'relu') (middle)
output1 = Dense(50, activation= 'relu') (output1)
output1 = Dense(20, activation= 'relu') (output1)
output1 = Dense(3, activation= 'relu') (output1)

output2 = Dense(100, activation= 'relu') (middle)
output2 = Dense(50, activation= 'relu') (output2)
output2 = Dense(20, activation= 'relu') (output2)
output2 = Dense(1, activation= 'relu') (output2)


model = Model(inputs = [input1,input2], outputs = [output1,output2])


# 컴파일, 훈련


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit([x1,x2],[y1,y2],epochs=400, batch_size=5,
callbacks = early_stopping)

# 평가 및 예측

loss = model.evaluate([x1,x2],[y1,y2])
print("loss, mae : " ,loss)

x1_pred = x1_pred.reshape(1,2,1)
x2_pred = x2_pred.reshape(1,3,1)
result1, result2 = model.predict([x1_pred,x2_pred])
print(result1)
print(result2)

'''
loss, mae :  [0.6670510172843933, 1.8083634131471626e-05, 0.6670329570770264, 0.003270816756412387, 0.6710585355758667]
[[54.0074]]
[[47.75568]]
'''