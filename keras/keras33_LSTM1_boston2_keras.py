#텐서플로우 데이터셋
#LSTM 으로 모델링
#Dense와 성능비교
#회귀모델


#sklearn 데이터셋
#LSTM 으로 모델링
#Dense와 성능비교
#회귀모델

#보스턴 집값 boston
#실습 : validation을 분리하여 전처리한 뒤 모델을 완성해라
# validation_split -> validation_data

import numpy as np
#1. data
from tensorflow.keras.datasets import boston_housing

(x_train,y_train),(x_test,y_test)= boston_housing.load_data()


#데이터 전처리(MinMax)
#x = x/711.         # 틀린놈.
# x = (x - 최소) / (최대 -최소)
#   = (x - np.min(x)) / (np.max(x) -np.min(x))



from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, 
                                train_size = 0.8, shuffle = True, random_state = 66)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape)    # 323,13
print(x_test.shape)     # 102,13

x_train = x_train.reshape(323,13,1)


#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(13,1))
lstm = LSTM(30, activation= 'relu') (input1)
dense1 = Dense(50, activation= 'relu') (lstm)
dense1 = Dense(40, activation= 'relu') (dense1)
dense1 = Dense(60, activation= 'relu') (dense1)
dense1 = Dense(35, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1, output1)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode = 'auto')

# compile, fit
model.compile(loss = 'mse', optimizer= 'adam', metrics= ['mae'])
model.fit(x_train, y_train, epochs = 200, batch_size= 10, 
        validation_data= (x_val,y_val), callbacks=early_stopping)

# evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1, verbose =1)
print("loss : ", loss)
x_test = x_test.reshape(102,13,1)
y_predict = model.predict(x_test)
y_pred = y_predict.reshape(1,-1)
print(y_pred)




