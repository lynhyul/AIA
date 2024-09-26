import numpy as np
import pandas as pd


# size = 1번(삼성전자주가)데이터

df = pd.read_csv('../data/csv/삼성전자.csv', index_col=0,header=0,encoding='CP949')
df1 = pd.read_csv('../data/csv/삼성전자0115.csv', index_col=0,header=0,encoding='CP949')
df.replace(',','',inplace=True, regex=True)
df1.replace(',','',inplace=True, regex=True)


#데이터 병합
df.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14']

df1.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
df.drop(['5','6','7','8','9','11','13','14'], axis='columns', inplace=True)
df1.drop(['5','6','7','8','9','10','11','13','15','16'], axis='columns', inplace=True)

df1 = df1.astype('float32')

df.columns = ['시가','고가','저가','종가','기관','외국계']
df1.columns=['시가','고가','저가','종가','기관','외국계']


df = df.astype('float32')
df1 = df1.astype('float32')

df = df.iloc[:662,:]
df1 = df1.iloc[:2,:]

df = df.sort_values(by=['일자'], axis=0)
df1 = df1.sort_values(by=['일자'], axis=0)

df= df.append(df1,ignore_index=False)

df_y = df.iloc[4:,:1] 
df_x = df.iloc[:-2,:]
df_x_pred = df.iloc[-4:,:]

print(df_x_pred)

x_data = df_x.to_numpy()
y_data = df_y.to_numpy()
x_pred = df_x_pred.to_numpy()  

print(x_data.shape) #664,7
print(y_data.shape) #664,1

size = 4
def split_x(seq, size):

    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x1 = split_x(x_data,size)
y1 = split_x(y_data,2)
x_pred = split_x(x_pred,4)

print(x1.shape) #(659,4,7)
print(x_pred.shape) # 1,4,7

# x1 = x1.reshape(659,6,4)
# y1 = y1.reshape(659,1,2)
# x_pred = x_pred.reshape(1,6,4)


# # 2번(코스피) 데이터
df2 = pd.read_csv('../data/csv/KODEX 코스닥150 선물인버스.csv', index_col=0,header=0,encoding='CP949')
df2.replace(',','',inplace=True, regex=True)
df2.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
df2.drop(['5','6'], axis='columns', inplace=True)
df2.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
df2 = df2.astype('float32')

df2 = df2.iloc[:664,[0,1,2,3,4,6]]
df2 = df2.sort_values(by=['일자'], axis=0)


df_y2 = df2.iloc[4:,:1] 
df_x2 = df2.iloc[:-2,:]


x2_data = df_x2.to_numpy()
y2_data = df_y2.to_numpy()



def split_x2(seq, size):

    aaa2 = []
    for a in range(len(seq) - size + 1):
        subset2 = seq[a : (a+size)]
        aaa2.append(subset2)
    return np.array(aaa2)

x2 = split_x2(x2_data,4)
y2 = split_x2(y2_data,2)

# x2 = x2.reshape(659,6,4)
# y2 = y2.reshape(659,1,2)

print(x2)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,test_size = 0.2, shuffle = True, random_state=101)
x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train,y1_train,train_size = 0.8)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,test_size = 0.2, shuffle = True, random_state=101)
x2_train, x2_val, y2_train, y2_val = train_test_split(x2_train,y2_train,train_size = 0.8)

x1_train = x1_train.reshape(x1_train.shape[0],24)
x1_test = x1_test.reshape(x1_test.shape[0],24)
x2_train = x2_train.reshape(x2_train.shape[0],24)
x2_test = x2_test.reshape(x2_test.shape[0],24)
x1_val = x1_val.reshape(x1_val.shape[0],24)
x2_val = x2_val.reshape(x2_val.shape[0],24)
x_pred = x_pred.reshape(x_pred.shape[0],24)

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

x1_train = x1_train.reshape(x1_train.shape[0],4,6)
x1_test = x1_test.reshape(x1_test.shape[0],4,6)
x2_train = x2_train.reshape(x2_train.shape[0],4,6)
x2_test = x2_test.reshape(x2_test.shape[0],4,6)
x1_val = x1_val.reshape(x1_val.shape[0],4,6)
x2_val = x2_val.reshape(x2_val.shape[0],4,6)
x_pred = x_pred.reshape(x_pred.shape[0],4,6)



 

from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, Dropout, Flatten,MaxPooling1D



input1 = Input(shape = (4,6))
# lstm1 = Conv1D(filters = 400, kernel_size = 2, activation= 'relu') (input1)
# dense1 = Conv1D(400,2, activation= 'relu') (lstm1)
lstm1 = LSTM(30, activation= 'relu') (input1)
dense1 = Dense(50, activation= 'relu') (lstm1)
dense1 = Flatten() (dense1)
dense1 = Dense(315, activation= 'relu') (dense1)


input2 = Input(shape = (4,6))
# lstm2 =Conv1D(filters = 400, kernel_size = 2, activation= 'relu') (input2)
# dense2 = Conv1D(400,2, activation= 'relu') (lstm2)
lstm2 = LSTM(30, activation= 'relu') (input2)
dense2 = Dense(50, activation= 'relu') (lstm2)
dense2 = Flatten() (dense2)
dense2 = Dense(250, activation= 'relu') (dense2)


from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1,dense2])
middle = Dense(310, activation= 'relu') (merge1)
middle = Dropout(0.2) (middle)
middle = Dense(200, activation= 'relu') (middle)

output1 = Dense(200, activation= 'relu') (middle)
output1 = Dense(200, activation= 'relu') (output1)
output1 = Dense(310, activation= 'relu') (output1)
output1 = Dense(2) (output1)



model = Model(inputs = [input1,input2], outputs = [output1])


# # 컴파일, 훈련


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train,x2_train],[y1_train,y2_train],epochs=300, batch_size=30, validation_data=([x1_val,x2_val],[y1_val,y2_val]),
callbacks = early_stopping)

# 평가 및 예측

loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
print("loss, mae : " ,loss)


result= model.predict([x_pred,x_pred])
print(result)
