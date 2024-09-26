#sklearn 데이터셋
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
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x= dataset.data
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

print(x_train.shape)    # 282,10
print(x_test.shape)     # 89,10


x_train = x_train.reshape(282,10,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(10,1))
lstm = LSTM(25, activation= 'relu',return_sequences=True) (input1)
lstm = LSTM(30, activation= 'relu') (lstm)
dense1 = Dense(100, activation= 'relu') (lstm)
dense1 = Dense(55, activation= 'relu') (dense1)
dense1 = Dense(25, activation= 'relu') (dense1)
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
y_predict = model.predict(x_test)
y_pred = y_predict.reshape(1,-1)
print(y_pred)






