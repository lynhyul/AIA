import numpy as np
import pandas as pd


# size = 2

df = pd.read_csv('../data/csv/삼성전자.csv', index_col=0,header=0,encoding='CP949')
df.replace(',','',inplace=True, regex=True)
df = df.astype('float32')


# 액분 전 데이터

df = df.iloc[:662,:]
df.drop(['등락률', '기관' ,'프로그램','신용비','개인','외인(수량)','외국계','외인비'], axis='columns', inplace=True)

# 상관 관계 50 먹이기 전 (기관, 프로그램)



# 액분 후 데이터 
'''
df_1 = df.iloc[:662,:]
df_2 = df.iloc[665:,:]
df = pd.concat([df_1,df_2])
df.iloc[662:,0:4] = df.iloc[662:,0:4]/50.0
df.iloc[662:,5:] = df.iloc[662:,5:]*50
df.drop(['등락률', '기관' ,'금액(백만)','신용비','프로그램','외인(수량)','외국계','외인비'], axis='columns', inplace=True)
'''
# 상관 관계 50 먹이기 후 (거래량, 개인)

df = df.sort_values(by=['일자'], axis=0)

print(df)

df_x = df.iloc[:,[0,1,2,4,5]]

df_x1 = df_x.iloc[:-1]
print(df_x1)

df_y = df.iloc[1:,[3]]

x_pred = df_x.iloc[-1:]

print(x_pred)

# df_y = df.iloc[:,[]]

#df_y = df.iloc[size:,[3]]
#df_x_pred = df.iloc[-size:,[0,1,2,4,5]]

x = df_x1.to_numpy()
y = df_y.to_numpy()
x_pred = x_pred.to_numpy()

print(x_pred)

# def split_x(seq, size):

#     aaa = []
#     for i in range(len(seq) - size + 1):
#         subset = seq[i : (i+size)]
#         aaa.append(subset)
#     return np.array(aaa)

# total_data = split_x(x_data,size)


# npy 저장
# np.save('../Study/samsung/samsung_data.npy', arr=total_data)



# x = total_data[:-1,:size, :-1]
# y = total_data[1:,size-1,-1:]
# x_pred = total_data[-1,:size,:-1]
# print(x_pred)
# print(x_pred.shape)

# # 상관 계수 시각화!
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()
x_pred = x_pred.reshape(-1,5)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, random_state=101)
x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
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
