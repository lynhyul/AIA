import numpy as np
import pandas as pd


x1 = np.load('../data/npy/삼성전자2.npy',allow_pickle=True)[0]
y1 = np.load('../data/npy/삼성전자2.npy',allow_pickle=True)[1]
x_pred = np.load('../data/npy/삼성전자2.npy',allow_pickle=True)[2]

x2 = np.load('../data/npy/코스닥.npy',allow_pickle=True)[0]



from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1,x2,y1,test_size = 0.2, shuffle = False)
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



input1 = Input(shape = (4,6))
# lstm1 = Conv1D(filters = 400, kernel_size = 2, activation= 'relu') (input1)
# dense1 = Conv1D(400,2, activation= 'relu') (lstm1)
lstm1 = LSTM(400, activation= 'relu') (input1)
dense1 = Dense(350, activation= 'relu') (lstm1)
dense1= Dropout(0.2) (dense1)
dense1 = Dense(315, activation= 'relu') (dense1)
dense1= Dropout(0.2) (dense1)
dense1 = Dense(325, activation= 'relu') (dense1)
dense1= Dropout(0.2) (dense1)
dense1 = Dense(335, activation= 'relu') (dense1)
dense1= Dropout(0.2) (dense1)

input2 = Input(shape = (4,6))
# lstm2 =Conv1D(filters = 400, kernel_size = 2, activation= 'relu') (input2)
# dense2 = Conv1D(400,2, activation= 'relu') (lstm2)
lstm2 = LSTM(400, activation= 'relu') (input2)
dense2 = Dense(30, activation= 'relu') (lstm2)
dense2 = Dense(300, activation= 'relu') (dense2)
dense2 = Dropout(0.2) (dense2)
dense2 = Dense(315, activation= 'relu') (dense2)
dense2 = Dropout(0.2) (dense2)
dense2 = Dense(330, activation= 'relu') (dense2)
dense2 = Dropout(0.2) (dense2)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1,dense2])
middle = Dense(310, activation= 'relu') (merge1)
middle = Dropout(0.2) (middle)
middle = Dense(200, activation= 'relu') (middle)



output1 = Dense(200, activation= 'relu') (middle)
output1 = Dense(2) (output1)



model = Model(inputs = [input1,input2], outputs = [output1])


# # 컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/삼성전자_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10, factor=0.5, verbose=1)
#n번까지 참았는데도 개선이없으면 50퍼센트 감축시키겠다.

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train,x2_train],y1_train,epochs=300, batch_size=30, validation_data=([x1_val,x2_val],[y1_val]),
callbacks = [early_stopping,cp,reduce_lr])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '/content/삼성전자_{epoch:02d}-{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

model.save('../data/h5/삼성전자3.h5')

# 평가 및 예측

loss = model.evaluate([x1_test,x2_test],y1_test, batch_size=1)
print("loss, mae : " ,loss)


result= model.predict([x_pred,x_pred])
print("시가(월요일,화요일) : ",result)

'''
before
loss, mae :  [14084309.0, 2939.058837890625]

after ad reduce_lr
loss, mae :  [3049858.5, 1379.7821044921875]
loss, mae :  [2949230.5, 1362.5491943359375
'''