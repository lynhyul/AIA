# 2개의 파일을 만드시오.
# 1.EarlyStopping을 적용하지 않은 최고의 모델 
# 2.EarlyStopping을 적용한 최고의 모델

#from tensorflow.keras.datasets import boston_housing
# 이걸로 만들어라!!

import numpy as np
#1. data
from tensorflow.keras.datasets import boston_housing


(x_train,y_train),(x_test,y_test)= boston_housing.load_data()

from sklearn.model_selection import train_test_split
(x_train,x_val , y_train, y_val) = train_test_split(x_train,y_train, train_size = 0.8)

print(np.max(x_train), np.min(x_train)) #  711 0
#print(dataset.DESCR)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(350, activation= 'relu') (input1)
dense1 = Dense(500, activation= 'relu') (dense1)
dense1 = Dense(1050, activation= 'relu') (dense1)
dense1 = Dense(2050, activation= 'relu') (dense1)
dense1 = Dense(1050, activation= 'relu') (dense1)
dense1 = Dense(600, activation= 'relu') (dense1)
dense1 = Dense(350, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1, output1)

# compile, fit
model.compile(loss = 'mse', optimizer= 'adam', metrics= ['mae'])


model.fit(x_train, y_train, epochs = 400, batch_size= 8, validation_data= (x_val,y_val))

# evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1, verbose =1)
print("loss : ", loss)
y_predict = model.predict(x_test)



#RMSE, R2

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test,y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)
