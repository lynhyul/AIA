#실습 validation data를 만들것!
#조건: train_test_split를 사용할 것

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x = np.arange(1,101)    #1~100
y = np.arange(1,101)    #1~100 // y=X (W=1)

#x_train = x[:60]                # 1 ~ 60
#x_val = x[60:80]                # 61 ~ 80
#x_test = x[80:]                 # 81 ~ 100

#y_train = y[:60]                # 1 ~ 60
#y_val = y[60:80]                # 61 ~ 80
#y_test = y[80:]                 # 81 ~ 100

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, \
                            train_size = 0.7, test_size =0.2,shuffle = True)
                            #train_size = 0.9, test_size =0.2,shuffle = True)
# shuffle = False 랜덤변수 셔플을 사용하지 않겠다.


print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#(70,)
#(20,)
#(70,)
#(20,)               
#train_size = 0.7로인해 train은 70만 나오는걸 확인, 10만큼의 리스트가 사라졌다.

'''
#2. model
model = Sequential()
model.add(Dense(20,input_dim=1,activation = 'relu' ))
for i in range(1,5) :
    model.add(Dense(10-i))
model.add(Dense(1))

#3. compile, train
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 200, validation_data=(x_val,y_val))

#4. evaluate, predict
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)


y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error      #mse
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))       #sqrt -> 루트를 씌우다
print("RMSE : ", RMSE (y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)

#loss :  0.09226157516241074
#mae :  0.09226157516241074
#RMSE :  0.10695013615047597
#R2 :  0.9999834032246222
'''