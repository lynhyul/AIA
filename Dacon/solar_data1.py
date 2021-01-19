import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf



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
data = train_x.to_numpy()
data = data.reshape(1095,48,6)
print(data.shape) # 1095,48,6

for a in range(81) :
    globals()['pred{}'.format(a)] = globals()['test_{}'.format(a)].to_numpy()
# print(pred80.shape) # 336,6

def split_xy(dataset, timesteps_x, timesteps_y):
    x, y = list(), list()
    
    for i in range(len(data)):
        x_end_number = i + timesteps_x
        y_end_number = x_end_number + timesteps_y
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

timesteps_x = 7
timesteps_y = 2

x, y = split_xy(data, timesteps_x, timesteps_y)
print(x.shape, y.shape) # (1087, 7, 48, 6) (1087, 2, 48, 6)

y = y.reshape(1087, 2*48*6)

print(x.shape)
for b in range(81) :
    globals()['pred{}'.format(b)]= (globals()['pred{}'.format(b)]).reshape(1,336,6)
print(x.shape)      # 695, 7, 48, 6
print(y.shape)      # 695, 7, 48, 6

# print(x.shape) # 52129, 336, 6
# print(y.shape) # 52129, 96, 6
# y = y.reshape(y.shape[0],96*6)
# # print(pred0.shape)  # 1,336,6


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, random_state=101)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8)

print(x_train.shape) # (33633, 6, 6)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2]*x_val.shape[3])
for b in range(81) :
    globals()['pred{}'.format(b)]= (globals()['pred{}'.format(b)]).reshape(1,336*6)

print(x_train.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
for b in range(81) :
    globals()['pred{}'.format(b)]= scaler.transform(globals()['pred{}'.format(b)])


x_train = x_train.reshape(x_train.shape[0],7,48,6)
x_test = x_test.reshape(x_test.shape[0],7,48,6)
x_val = x_val.reshape(x_val.shape[0],7,48,6)
for b in range(81) :
    globals()['pred{}'.format(b)]= (globals()['pred{}'.format(b)]).reshape(1,336,1,6)



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv2D, Reshape, Flatten
inputs = Input(shape=(7,48,6))
dense1 = Conv2D(512, 2, padding='same')(inputs)
dense1 = Flatten()(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(16)(dense1)
outputs = Dense(2*48*6)(dense1)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

# model = Model(input1, output1)

# #3. compile fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/태양광_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience=40)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=20, factor=0.5, verbose=1)
#n번까지 참았는데도 개선이없으면 50퍼센트 감축시키겠다.

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train,epochs=300, batch_size=120, validation_data=(x_val,y_val),
callbacks = [early_stopping,cp,reduce_lr])


loss = model.evaluate(x_test,y_test, batch_size=1)
print("loss : ",loss)


for b in range(81) :
    globals()['result{}'.format(b)] = (model.predict(globals()['pred{}'.format(b)])).reshape(96,6)
    globals()['result{}'.format(b)] = (model.predict(globals()['pred{}'.format(b)])).reshape(96,6)
    # np.savetxt('../data/csv/태양광{}test.csv'.format(b),globals()['result{}'.format(b)],delimiter=",")
    # globals()['result{}'.format(b)] = np.around((globals()['pred{}'.format(b)]),3)

# print("\n",result78)   # 96,6
'''
loss :  [25677.662109375, 0.010015263222157955]
'''


