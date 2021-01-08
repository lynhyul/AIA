# loss, val_loss 

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

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size = 0.8)

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
dense1 = Dense(50, activation= 'relu') (dense1)
dense1 = Dense(10, activation= 'relu') (dense1)
dense1 = Dense(30, activation= 'relu') (dense1)
dense1 = Dense(10, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1, output1)

# compile, fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=15, mode = 'auto')
model.compile(loss = 'mse', optimizer= 'adam', metrics= ['mae'])
hist = model.fit(x_train, y_train, epochs = 1500, batch_size= 8, 
                validation_data= (x_val,y_val), callbacks=[es])

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

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
# plt.plot(hist.history['acc'])       #회귀모델이기때문에 acc측정이 힘들다
# plt.plot(hist.history['val_acc'])
plt.title('loss & val_loss')
plt.ylabel('loss & val_loss')
plt.xlabel('epoch')
plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
plt.show()

'''
loss :  [5.088789463043213, 1.755423665046692]
'''