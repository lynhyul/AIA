#sklearn 데이터셋
#LSTM 으로 모델링
#Dense와 성능비교
#회귀모델

#보스턴 집값 boston
#실습 : validation을 분리하여 전처리한 뒤 모델을 완성해라
# validation_split -> validation_data

import numpy as np
#1. data
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)      # (506,13) -> calam 13, scalar 506
print(y.shape)      # (506,)
print("==================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 711.0.0.0
print(dataset.feature_names)
#print(dataset.DESCR)

#데이터 전처리(MinMax)
#x = x/711.         # 틀린놈.
# x = (x - 최소) / (최대 -최소)
#   = (x - np.min(x)) / (np.max(x) -np.min(x))

print(np.max(x[0]))

print(np.max(x), np.min(x))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle = True,
                                        random_state = 101)

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size = 0.8)

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
lstm = LSTM(100, activation= 'relu') (input1)
dense1 = Dense(100, activation= 'relu') (lstm)
dense1 = Dense(25, activation= 'relu') (dense1)
dense1 = Dense(35, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1, output1)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=15, mode = 'auto')

# compile, fit
model.compile(loss = 'mse', optimizer= 'adam', metrics= ['mae'])
model.fit(x_train, y_train, epochs = 300, batch_size= 15, 
        validation_data= (x_val,y_val), callbacks=early_stopping)

# evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1, verbose =1)
print("loss : ", loss)
x_test = x_test.reshape(102,13,1)
y_predict = model.predict(x_test)
y_pred = y_predict.reshape(1,-1)
print(y_pred)



'''
loss :  [26.765457153320312, 3.5205321311950684]
[[41.178093 30.044462 16.367346 16.275143 30.337206 26.367424 38.296215
  13.058338 40.597137 11.108187 29.411507 15.230773 20.114578 23.132296
  22.701555 22.226746 11.962127 32.535282 26.534971 24.750404 13.24699
  20.2538   25.402903 32.7792   32.879593 19.894571 27.052822 18.425312
  30.397234 34.756912 21.756758 21.055563 38.092255 52.652225 26.967005
  22.145065 14.368285 20.854097 10.063287 33.779694 21.313242 25.51902
  34.710262 14.119798 18.113348 24.975946 30.66311  16.603325 25.879469
  26.200651 34.06462  42.4936   21.977106 17.863503 34.990864  9.37868
  18.960863 16.811062 20.718061 18.737139 31.390497 11.442839 31.725689
  21.296736 11.930711 23.138227 22.930452 21.221792 13.928755 21.136944
  22.675566 24.714056 17.232449 20.73631  25.901152 11.561636 49.733555
  13.770211 34.882114 14.263366 18.609076 20.93358  28.412256 15.023518
  14.639908 21.310902 21.315966 26.511173 21.498835 19.78711  12.966578
  16.10858  34.386612 31.871168 10.306447 35.29685  15.099876 30.113544
  12.965069 21.431469 33.16408  20.65047 ]]
  '''


