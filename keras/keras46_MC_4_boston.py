#sklearn 데이터셋
#LSTM 으로 모델링
#Dense와 성능비교
#회귀모델

#보스턴 집값 boston
#실습 : validation을 분리하여 전처리한 뒤 모델을 완성해라
# validation_split -> validation_data

import numpy as np
#1. data
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)      # (506,13) -> calam 13, scalar 506
print(y.shape)      # (506,)
print("==================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 711.0.0.0
print(dataset.feature_names)
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

print(x_train.shape)    # 323,13
print(x_test.shape)     # 102,13

x_train = x_train.reshape(323,13,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape=(13,1))
lstm = LSTM(400, activation= 'relu',return_sequences=True) (input1)
lstm = LSTM(300, activation= 'relu') (lstm)
dense1 = Dense(100, activation= 'relu') (lstm)
dense1 = Dense(65, activation= 'relu') (dense1)
dense1 = Dense(35, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1, output1)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './modelCheckpoint/k46_boston_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience= 5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')

# compile, fit
model.compile(loss = 'mse', optimizer= 'adam', metrics= ['acc'])
hist = model.fit(x_train, y_train, epochs = 200, batch_size= 10, 
        validation_data= (x_val,y_val), callbacks=[early_stopping,cp])

# evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1, verbose =1)
print("loss : ", loss)
y_predict = model.predict(x_test)
y_pred = y_predict.reshape(1,-1)
print(y_pred)




import matplotlib.pyplot as plt

plt.figure(figsize= (10,6)) # 판을 깔아준다.

plt.subplot(2, 1, 1)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 1번째   
plt.plot(hist.history['loss'], marker='.', c='red',label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue',label='val_loss')
plt.grid()

plt.title('Cost loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 2번째   
plt.plot(hist.history['acc'], marker='.', c='red',label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue',label='val_acc')
plt.grid()

plt.title('Accuarcy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.show()