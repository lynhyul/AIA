
# loss, val_loss 

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x= dataset.data
y = dataset.target



print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)

print(np.max(x), np.min(y))
print(dataset.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(dataset.DESCR)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle = True,
                                        random_state = 101)

x_train,x_val,y_train, y_val = train_test_split(x_train,y_train,train_size=0.7)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(10,))
dense1 = Dense(400, activation= 'relu') (input1)
dense1 = Dense(500, activation= 'relu') (dense1)
dense1 = Dense(70, activation= 'relu') (dense1)
dense1 = Dense(70, activation= 'relu') (dense1)
dense1 = Dense(100, activation= 'relu') (dense1)
dense1 = Dense(35, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1, output1)

# compile, fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience = 10, mode ='auto')

model.compile(loss = 'mse', optimizer= 'adam', metrics= ['mae'])
hist = model.fit(x_train, y_train, epochs = 1000, batch_size= 10, validation_data=(x_val,y_val),
                                        callbacks=[early_stopping])

# evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1, verbose =1)
print("loss : ", loss)
y_predict = model.predict(x_test)


#RMSE, R2

# from sklearn.metrics import mean_squared_error, r2_score
# def RMSE(y_test, y_predict) :
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", RMSE(y_test,y_predict))

# r2 = r2_score(y_test, y_predict)
# print("R2 : ",r2)

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
