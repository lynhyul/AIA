import pandas as pd
import numpy as np
import os
import glob
import random
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M


train = pd.read_csv('../data/csv/train/train.csv',index_col=False)
# test = pd.read_csv('../data/csv/test/test.csv',index_col=False)
# modelpath = '../data/modelcheckpoint/태양광_{epoch:02d}-{val_loss:.4f}.hdf5'
for i in range(81) :
    filepath = '../data/csv/test/{}.csv'.format(i)
    globals()['test{}'.format(i)] = pd.read_csv(filepath,index_col=False)
    globals()['test_{}'.format(i)] = globals()['test{}'.format(i)].iloc[:,3:]
# sub = pd.read_csv('../data/csv/sample_submission.csv',index_col=False)

train_x = train.iloc[:,3:]


'''
  DHI  DNI   WS     RH   T  TARGET
0        0    0  1.5  69.08 -12     0.0
1        0    0  1.5  69.06 -12     0.0
'''
x_data = train_x.to_numpy()
for a in range(81) :
    globals()['pred{}'.format(a)] = globals()['test_{}'.format(a)].to_numpy()
# print(pred80.shape) # 336,6

def split_xy2(dataset,time_steps,y_column) :
    x, y = list(),list()
    for i in range(len(dataset)) :
        x_end_number = i+ time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

time_steps = 336
y_column = 96
x, y = split_xy2(x_data, time_steps, y_column)
print(x.shape)
# for b in range(81) :
#     globals()['pred{}'.format(b)]= (globals()['pred{}'.format(b)]).reshape(1,336,6)

# print(x.shape) # 52129, 336, 6
# print(y.shape) # 52129, 96, 6
# y = y.reshape(y.shape[0],96*6)
# # print(pred0.shape)  # 1,336,6


# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, random_state=101)
# x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8)

# print(x_train.shape) # (33633, 6, 6)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
# x_val = x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
# for b in range(81) :
#     globals()['pred{}'.format(b)]= (globals()['pred{}'.format(b)]).reshape(1,336*6)

# print(x_train.shape)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)
# for b in range(81) :
#     globals()['pred{}'.format(b)]= scaler.transform(globals()['pred{}'.format(b)])


# x_train = x_train.reshape(x_train.shape[0],336,6)
# x_test = x_test.reshape(x_test.shape[0],336,6)
# x_val = x_val.reshape(x_val.shape[0],336,6)
# for b in range(81) :
#     globals()['pred{}'.format(b)]= (globals()['pred{}'.format(b)]).reshape(1,336,6)


# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, Dropout, Flatten,MaxPooling1D


# model = Sequential()
# model.add(Conv1D(filters = 20, kernel_size = 2, input_shape=(336,6)))
# # model.add(MaxPooling1D(pool_size=2))
# # model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(512, activation= 'relu'))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dense(96*6))


# model.summary()

# # model = Model(input1, output1)

# # #3. compile fit
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# modelpath = '../data/modelcheckpoint/태양광_{epoch:02d}-{val_loss:.4f}.hdf5'
# early_stopping = EarlyStopping(monitor='val_loss', patience=15)
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')

# reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=8, factor=0.5, verbose=1)
# #n번까지 참았는데도 개선이없으면 50퍼센트 감축시키겠다.

# model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
# model.fit(x_train,y_train,epochs=5, batch_size=120, validation_data=(x_val,y_val),
# callbacks = [early_stopping,cp,reduce_lr])

# # from tensorflow.keras.callbacks import EarlyStopping
# # early_stopping = EarlyStopping(monitor='loss', patience= 5, mode = 'auto')

# # model.compile(loss = 'mse', optimizer='adam', metrics=['acc'])
# # model.fit(x_train,y_train, epochs=100, batch_size=2, validation_data=(x_val, y_val),  
# #                                      callbacks = early_stopping)

# #4. evaluate , predict
# # y_train = y.train.reshape(y_train.shape[0],96,6)
# # y_test = y.test.reshape(y_test.shape[0],96,6)

# loss = model.evaluate(x_test,y_test, batch_size=1)
# print("loss : ",loss)

# # y = y.reshape(y.shape[0],96,6)
# # for b in range(81) :
# #     globals()['pred{}'.format(b)]= (globals()['pred{}'.format(b)]).reshape(1,336,6)
# result = model.predict(pred0)
# result = result.reshape(-1,96,6)
# print("\n",result)
# '''
# loss :  [25677.662109375, 0.010015263222157955]
# '''


# # # x_pred = x_pred.reshape(1,5,1)
# # # y_predict = model.predict(x_pred)
# # # print("y_predcit: ",y_predict)