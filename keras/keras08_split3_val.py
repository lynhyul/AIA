from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x = np.arange(1,101)    #1~100
y = np.arange(101,201)    #101~200 // y=1x+100

#x_train = x[:60]                # 1 ~ 60
#x_val = x[60:80]                # 61 ~ 80
#x_test = x[80:]                 # 81 ~ 100

#y_train = y[:60]                # 1 ~ 60
#y_val = y[60:80]                # 61 ~ 80
#y_test = y[80:]                 # 81 ~ 100

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size =0.8,shuffle = True)
# shuffle = False 랜덤변수 셔플을 사용하지 않겠다.
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# train_test_split(x,y, train_size =0.6), x데이터와 y데이터를 (x:y=)6:4 분리시키겠다.

#2. model
model = Sequential()
model.add(Dense(20,input_dim=1,activation = 'relu' ))
for i in range(1,5) :
    model.add(Dense(10-i))
model.add(Dense(1))

#3. compile, train
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 200, validation_split=0.2)

#4. evaluate, predict
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)


y_predict = model.predict(x_test)
print(y_predict)

