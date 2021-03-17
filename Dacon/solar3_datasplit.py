import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.backend import mean, maximum



train = pd.read_csv('../data/csv/train/train.csv',index_col=False)
# test = pd.read_csv('../data/csv/test/test.csv',index_col=False)
# modelpath = '../data/modelcheckpoint/태양광_{epoch:02d}-{val_loss:.4f}.hdf5'
df_test = []
for i in range(81) :
    filepath = '../data/csv/test/{}.csv'.format(i)
    # globals()['test{}'.format(i)] = pd.read_csv(filepath,index_col=False)
    df = pd.read_csv(filepath,index_col=False)
    df = df[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    df = df.iloc[:-48,:] 
    df_test.append(df)
X_test = pd.concat(df_test)
print(X_test)

temp = train.copy()
temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]      
temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
temp = temp.dropna()

# train_x = train.iloc[:,3:-1]
# print(temp.tail())

x = temp.iloc[:-96,:-2]
y = temp.iloc[:-96,-2:]

x = x.to_numpy()
y = y.to_numpy()
pred = X_test.to_numpy()
# print(pred.shape)   #27216,7

# print(x.shape)  #(52464, 7)

x = x.reshape(1093,48,7)
y = y.reshape(1093,48,2)
print(pred)
print(pred.shape)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, random_state=101)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8)

print(x_train.shape) #669,48,7

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1]*x_val.shape[2])
pred = pred.reshape(486,336)


print(x_train.shape)    #669, 336
print(pred.shape)
print(x_val.shape)
print(x_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
pred = scaler.transform(pred)


x_train = x_train.reshape(x_train.shape[0],48,7)
x_test = x_test.reshape(x_test.shape[0],48,7)
x_val = x_val.reshape(x_val.shape[0],48,7)
pred = pred.reshape(486,48,7)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/태양광_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=40)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss',patience=20, factor=0.5, verbose=1)


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Reshape, Flatten,Reshape,LeakyReLU

inputs = Input(shape=(48,7))
dense1 = Conv1D(512, 2, padding='same',activation='relu')(inputs)
dense1 = LeakyReLU(alpha=0.5) (dense1)
dense1 = Conv1D(216, 2, padding='same',activation='relu')(dense1)
dense1 = LeakyReLU(alpha=0.5) (dense1)
# dense1 = LeakyReLU(alpha=0.5) (dense1)
dense1 = Conv1D(128, 2, padding='same',activation='relu')(dense1)
dense1 = LeakyReLU(alpha=0.5) (dense1)
# dense1 = Conv1D(64, 2,activation='relu')(dense1)
dense1 = Flatten()(dense1)
dense1 = Dense(256,activation='relu')(dense1)
dense1 = LeakyReLU(alpha=0.5) (dense1)
# dense1 = Dense(312,activation='relu')(dense1)
# dense1 = LeakyReLU(alpha=0.5) (dense1)
dense1 = Dense(48*2)(dense1) 
outputs = Reshape((48,2))(dense1)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

#3. compile fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/태양광_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=40)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss',patience=20, factor=0.5, verbose=1)
# n번까지 참았는데도 개선이없으면 50퍼센트 감축시키겠다.
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
# cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/dacon%d.hdf5',monitor='val_loss', save_best_only=True)
model.fit(x_train,y_train,epochs= 500, batch_size=10, validation_data=(x_val,y_val),callbacks=[cp,es,lr])

loss = model.evaluate(x_test,y_test, batch_size=1)
y_pred = model.predict(pred)
y_pred = np.round(y_pred,2)

print("loss : ", loss)
print("predict : \n", y_pred[:,:,:1]) # day 1 Target
print("predict : \n", y_pred[:,:,1:]) # day 2 Target
y_pred = np.append(y_pred[:,:,:1],y_pred[:,:,1:],axis=0)
print(y_pred.shape) # 972, 48, 1




def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)
q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
model1 = Sequential()
for q in q_lst:  
  model1.compile(loss=lambda y,pred: quantile_loss(q,y,y_pred), optimizer='adam')
  model1.fit(x_train,y_train, epochs =5, batch_size=1,validation_split=0.2)
  q_pred = model1.predict(pred)
print(q_pred) # 567,48,7
print(q_pred.shape)
q_pred.to_csv('../data/csv/submit5.csv')
'''
loss :  [163.5534210205078, 6.690919399261475]
'''




# submission = pd.read_csv('../data/csv/sample_submission.csv',index_col=False)

# d = []  
# for l in range(9):
#     cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/dacon%d.hdf5'%l,monitor='val_loss', save_best_only=True)
#     model.fit(x_train,y_train,epochs= 400, validation_data=(x_val,y_val), batch_size =30, callbacks = [es,cp,lr])
# df_data = pd.DataFrame(d)
# print(df_data)
