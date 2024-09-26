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
print("results(mse, mae)= ",results)

#np.sqrt(result[0])         # [0] mse  / [1] mae


y_predict = model.predict(x_test)          #x_pred를 넣어서 얻은 결과값은 y_predict
#print("y_predict :", y_predict)

#사이킷런
from sklearn.metrics import mean_squared_error      #mse
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))       #sqrt -> 루트를 씌우다
print("RMSE : ", RMSE (y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)