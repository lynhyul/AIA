
#EarlyStopping

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(13,))
dense1 = Dense(350, activation= 'relu') (input1)
dense1 = Dense(500, activation= 'relu') (dense1)
dense1 = Dropout(0.2) (dense1)
dense1 = Dense(1050, activation= 'relu') (dense1)
dense1 = Dense(2050, activation= 'relu') (dense1)
dense1 = Dropout(0.4) (dense1)
dense1 = Dense(1050, activation= 'relu') (dense1)
dense1 = Dropout(0.6) (dense1)
dense1 = Dense(600, activation= 'relu') (dense1)
dense1 = Dense(350, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1, output1)

# compile, fit
model.compile(loss = 'mse', optimizer= 'adam', metrics= ['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience = 10, mode ='auto')

model.fit(x_train, y_train, epochs = 2000, batch_size= 8, validation_split = 0.2,
                                        callbacks=[early_stopping])

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
loss :  [11.441758155822754, 2.3566792011260986]
RMSE :  3.382566917372662
R2 :  0.8615085888867398
'''