import numpy as np
import pandas as pd
import re

df = pd.read_csv('../data/csv/삼성전자.csv',index_col=0,header=0, encoding='cp949')

df = df.fillna(method = 'pad')

df['시가'] = df['시가'].str.replace(',','').astype('int64')
df['고가'] = df['고가'].str.replace(',','').astype('int64')
df['저가'] = df['저가'].str.replace(',','').astype('int64')
df['종가'] = df['종가'].str.replace(',','').astype('int64')
df['거래량'] = df['거래량'].str.replace(',','').astype('int64')
df['기관'] = df['기관'].str.replace(',','').astype('int64')

# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns
# sns.set(font_scale = 1.2)
# sns.heatmap(data=df.corr(), square =True, annot=True, cbar = True)
# # plt.rcParams['axes.unicode_minus'] = False 
# sns.set(font = 'Malgun Gothic', rc= {'axes.unicode_minus':False},style='darkgrid')
# plt.show()

df1 = df.to_numpy(dtype=None, copy=False)
# print(df1)

# print(df2)



print(df1[:10,:7])
'''
[[891000 892000 866000 867000 -2.36]
 [853000 875000 853000 875000 0.92]
 [889000 916000 886000 916000 4.69]
 [930000 937000 918000 928000 1.31]
 [924000 927000 902000 904000 -2.59]]
 '''
df3 = np.delete(df1,3,1)
df2 = df3[::-1,:5]
df1 = df1[::-1,:5]
print(df2)

def x_dataset(start,end) :
    aaa = []
    for i in range(start,end) :
        subset = df2[i:1+i,:]
        aaa.append(subset)
    return np.array(aaa)
print(x_dataset(1738,2399).shape) 
print(x_dataset(1738,2399)) 
x_dataset = x_dataset(1738,2399).reshape(661,5)



def y_dataset(start,end) :
    bbb = []
    for a in range(start,end) :
        subset = df1[a+1:a+2,3:4]

        bbb.append(subset)
    return np.array(bbb)

print(y_dataset(1738,2399).shape)  # 661,1
print(y_dataset(1738,2399))
y_dataset = y_dataset(1738,2399).reshape(661,1)

x = np.asarray(x_dataset).astype(np.float32)
y = np.asarray(y_dataset).astype(np.float32)

# print(y[0:2])
# print(y)

x_pred = np.array([89800,91200,89100,-0.99,34161101])
x_pred = x_pred.reshape(1,5)
print(x_pred)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, shuffle = True, random_state=101)
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

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,LSTM

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, Dropout, Flatten,MaxPooling1D


model = Sequential()
model.add(Conv1D(filters = 400, kernel_size = 2, input_shape=(5,1)))
model.add(Conv1D(400,2))
model.add(Conv1D(300,2))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(310, activation= 'relu'))
model.add(Dense(315, activation= 'relu'))
model.add(Dense(1))
# model = Sequential()
# model.add(LSTM(410, activation='relu', input_shape=(5,1)))
# model.add(Dense(250, activation= 'relu'))
# model.add(Dense(150, activation= 'relu'))
# model.add(Dense(150, activation= 'relu'))
# model.add(Dense(150, activation= 'relu'))
# model.add(Dense(30, activation= 'relu'))
# model.add(Dense(1))

model.summary()

# model = Model(input1, output1)

#3. compile fit

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 30, mode = 'auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=500, batch_size=30, validation_data=(x_val, y_val),  
                                     callbacks = early_stopping)

#4. evaluate , predict



loss = model.evaluate(x_test,y_test, batch_size=1)
print("loss : ",loss)


x_pred = x_pred.reshape(1,5,1)
y_predict = model.predict(x_pred)
print("y_predcit: ",y_predict)


'''
loss :  [923418.0, 0.0]
y_predcit:  [[89802.12]]
'''



