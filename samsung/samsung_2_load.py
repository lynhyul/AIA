import numpy as np
import pandas as pd
import re

import numpy as np

x = np.load('../data/npy/samsung_2.npy',allow_pickle=True)[0]
y = np.load('../data/npy//samsung_2.npy',allow_pickle=True)[1]
x_pred = np.load('../data/npy//samsung_2.npy',allow_pickle=True)[2]


# print(x)
x_pred = x_pred.reshape(-1,5)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, random_state=101)
x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)


x_train = x_train.reshape(x_train.shape[0],5,1)
x_val = x_val.reshape(x_val.shape[0],5,1)
x_test = x_test.reshape(x_test.shape[0],5,1)

print(x_test.shape) 
print(x_val.shape)  


print(x_train.shape)    


from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, Dropout, Flatten,MaxPooling1D





model = load_model('../data/h5/samsung_model1.h5')
# model.load_weights('../data/h5/samsung_weight1.h5')

x_pred = x_pred.reshape(1,5,1)
y_predict = model.predict(x_pred)

#4-2. evaluate , predict
result = model.evaluate(x_test,y_test, batch_size=32)
print("가중치_loss : ", result[0])
print("예측값 : ", y_predict)


'''
가중치_loss :  2164551.75
예측값 :  [[89640.28]]
'''



