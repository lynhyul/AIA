#sklearn 데이터셋
#LSTM 으로 모델링
#Dense와 성능비교
#이중분류



import numpy as np
#1. data
from sklearn.datasets import load_breast_cancer

#1. data

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target


print(x.shape)      # (506,13) -> calam 13, scalar 506
print(y.shape)      # (506,)
print("==================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 711.0.0.0
print(datasets.feature_names)
#print(dataset.DESCR)

#데이터 전처리(MinMax)
#x = x/711.         # 틀린놈.
# x = (x - 최소) / (최대 -최소)
#   = (x - np.min(x)) / (np.max(x) -np.min(x))

print(np.max(x[0]))

print(np.max(x), np.min(x))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle = True,
                                        random_state = 101)

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size = 0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape)    # 364,30
print(x_test.shape)     # 114,30


x_train = x_train.reshape(364,30,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(30,1))
lstm = LSTM(25, activation= 'relu',return_sequences=True) (input1)
lstm = LSTM(30, activation= 'relu') (lstm)
dense1 = Dense(100, activation= 'relu') (lstm)
dense1 = Dense(55, activation= 'relu') (dense1)
dense1 = Dense(25, activation= 'relu') (dense1)
output1 = Dense(1, activation='sigmoid') (dense1)

model = Model(input1, output1)

#compile, fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 5, mode = 'auto')
#model.compile(loss = 'mean_squared_error', optimizer='adam', metrics =['accuracy'])
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics =['accuracy','mae'])
model.fit(x_train,y_train, epochs = 200, batch_size=15, validation_split = 0.3, 
                            callbacks = early_stopping)

loss = model.evaluate(x_test,y_test, batch_size=1)
print("[loss, accuracy, mae] : ",loss)


y_predict = model.predict(x_test[0:15])
y_pred = list(map(int,np.round(y_predict,0)))
y_predict = np.transpose(y_predict)
#y_predict = np.where(y_predict>=0.5,1,y_predict)
y_pred = np.transpose(y_pred)
print(y_predict)
print("predict = ",y_pred)
print("target = ",y_test[0:15])






