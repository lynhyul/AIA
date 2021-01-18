import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


x1 = np.load('../data/npy/삼성전자2.npy',allow_pickle=True)[0]
y1 = np.load('../data/npy/삼성전자2.npy',allow_pickle=True)[1]
x_pred = np.load('../data/npy/삼성전자2.npy',allow_pickle=True)[2]

x2 = np.load('../data/npy/코스닥.npy',allow_pickle=True)[0]



from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1,x2,y1,test_size = 0.2, shuffle = True, random_state=101)
x1_train, x1_val,x2_train,x2_val, y1_train, y1_val = train_test_split(x1_train,x2_train,y1_train,train_size = 0.8)



x1_train = x1_train.reshape(x1_train.shape[0],24)
x1_test = x1_test.reshape(x1_test.shape[0],24)
x2_train = x2_train.reshape(x2_train.shape[0],24)
x2_test = x2_test.reshape(x2_test.shape[0],24)
x1_val = x1_val.reshape(x1_val.shape[0],24)
x2_val = x2_val.reshape(x2_val.shape[0],24)
x_pred = x_pred.reshape(x_pred.shape[0],24)
# x2_pred = x2_pred.reshape(x2_pred.shape[0],24)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_val = scaler.transform(x1_val)
x_pred = scaler.transform(x_pred)

scaler.fit(x2_train)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)
x2_val = scaler.transform(x2_val)
# x2_pred = scaler.transform(x2_pred)

x1_train = x1_train.reshape(x1_train.shape[0],4,6)
x1_test = x1_test.reshape(x1_test.shape[0],4,6)
x2_train = x2_train.reshape(x2_train.shape[0],4,6)
x2_test = x2_test.reshape(x2_test.shape[0],4,6)
x1_val = x1_val.reshape(x1_val.shape[0],4,6)
x2_val = x2_val.reshape(x2_val.shape[0],4,6)
x_pred = x_pred.reshape(x_pred.shape[0],4,6)
# x2_pred = x2_pred.reshape(x2_pred.shape[0],4,6)




from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, Dropout, Flatten,MaxPooling1D


# # 컴파일, 훈련


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




model = load_model('../data/h5/삼성전자3.h5')

# 평가 및 예측

loss = model.evaluate([x1_test,x2_test],y1_test)
print("loss, mae : " ,loss)


result= model.predict([x_pred,x_pred])
print("시가(월요일,화요일) : ",result)
