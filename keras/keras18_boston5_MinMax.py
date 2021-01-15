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
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, shuffle = True,
                                        random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size = 0.2)


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
model.fit(x_train, y_train, epochs = 1500, batch_size= 15, validation_data= (x_val,y_val))

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


'''
전처리 전
loss :  [12.677449226379395, 2.4839680194854736]
RMSE :  3.560540587793695
R2 :  0.848324913169763

전처리(모든 x의 컬럼에 대해서 동일하게 711로 나눠준것 = 잘못된 방식) 후
loss :  [11.792983055114746, 2.47595477104187]
RMSE :  3.4340910494922112
R2 :  0.8589068329892685

MinMaxScalar (validation MinMax x)
loss :  [10.03169059753418, 2.1911513805389404]
RMSE :  3.1672844431269938
R2 :  0.8785760989238386

MinMaxScalar(validation MinMax o)
loss :  [5.4402289390563965, 1.7170753479003906]
RMSE :  2.332429844313774
R2 :  0.9341512969036383
'''

