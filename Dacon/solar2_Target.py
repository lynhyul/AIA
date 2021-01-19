import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('../data/csv/train/train.csv',index_col=False)
# test = pd.read_csv('../data/csv/test/test.csv',index_col=False)
# modelpath = '../data/modelcheckpoint/태양광_{epoch:02d}-{val_loss:.4f}.hdf5'
df_test = []
for i in range(81) :
    filepath = '../data/csv/test/{}.csv'.format(i)
    # globals()['test{}'.format(i)] = pd.read_csv(filepath,index_col=False)
    df = pd.read_csv(filepath,index_col=False)
    df = df[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']] 
    df_test.append(df)
X_test = pd.concat(df_test)
print(X_test)

temp = train.copy()
temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]      
temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
temp = temp.dropna()

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# train_x = train.iloc[:,3:-1]
print(temp.tail())

x = temp.iloc[:-96,:-2]
y = temp.iloc[:-96,-2:]

x = x.to_numpy()
y = y.to_numpy()

pred = X_test.to_numpy()
# print(pred.shape)   #27216,7

# print(x.shape)  #(52464, 7)

x = x.reshape(1093,48,7)
y = y.reshape(1093,48,2)

# x = x.reshape(x.shape[0]/48,48,2)
# y = y.reshape(y.shape[0]/48,48,2)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, random_state=101)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8)

print(x_train.shape) 

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])



print(x_train.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)



x_train = x_train.reshape(x_train.shape[0],48,7)
x_test = x_test.reshape(x_test.shape[0],48,7)
x_val = x_val.reshape(x_val.shape[0],48,7)
pred = pred.reshape(567,48,7)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Reshape, Flatten,Reshape
inputs = Input(shape=(48,7))
dense1 = Conv1D(512, 2, padding='same',activation='relu')(inputs)
dense1 = Conv1D(216, 2, padding='same',activation='relu')(inputs)
dense1 = Conv1D(128, 2, padding='same',activation='relu')(inputs)
dense1 = Flatten()(dense1)
dense1 = Dense(64,activation='relu')(dense1)
dense1 = Dense(32,activation='relu')(dense1)
dense1 = Dense(48*2,activation='relu')(dense1)
dense1 = Reshape((48,2))(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

# model = Model(input1, output1)

# #3. compile fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/태양광_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=40)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')

lr = ReduceLROnPlateau(monitor='val_loss',patience=20, factor=0.5, verbose=1)
#n번까지 참았는데도 개선이없으면 50퍼센트 감축시키겠다.

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
# model.fit(x_train,y_train,epochs=300, batch_size=120, validation_split=0.2,
# callbacks = [es,cp,lr])

# loss = model.evaluate(x_test,y_test, batch_size=1)
# print("loss : ",loss)



d = []  
for l in range(9):
    cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/dacon%d.hdf5'%l,monitor='val_loss', save_best_only=True)
    model.fit(x,y,epochs= 300, validation_split=0.2, batch_size =120, callbacks = [es,cp,lr])
    y_pred = model.predict(pred)
    y_pred = y_pred.reshape(27216,2)
    d.append(y_pred)
d = np.array(d)
df_data = pd.DataFrame(d)
for i in range(9) :
    df_data = df.quantile(q = ((i+1)/10.),axis = 0)[0]
print(df_data)


# # submit 파일에 데이터들 덮어 씌우기!!
# for i in range(81):
#     for j in range(2):
#         for k in range(48):
#             df = pd.DataFrame(d[i,j,k])
#             for l in range(9):
#                 df_sub.iloc[[i*96+j*48+k],[l]] = df.quantile(q = ((l+1)/10.),axis = 0)[0]

# df_sub.to_csv('../data/csv/submit.csv')




# for b in range(81) :
#     globals()['result{}'.format(b)] = (model.predict(globals()['pred{}'.format(b)])).reshape(96,6)
#     globals()['result{}'.format(b)] = (model.predict(globals()['pred{}'.format(b)])).reshape(96,6)
#     # np.savetxt('../data/csv/태양광{}test.csv'.format(b),globals()['result{}'.format(b)],delimiter=",")
#     # globals()['result{}'.format(b)] = np.around((globals()['pred{}'.format(b)]),3)

# # print("\n",result78)   # 96,6
# '''
# loss :  [25677.662109375, 0.010015263222157955]
