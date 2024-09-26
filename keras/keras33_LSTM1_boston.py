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

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode = 'auto')

# compile, fit
model.compile(loss = 'mse', optimizer= 'adam', metrics= ['mae'])
model.fit(x_train, y_train, epochs = 200, batch_size= 10, 
        validation_data= (x_val,y_val), callbacks=early_stopping)

# evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1, verbose =1)
print("loss : ", loss)
y_predict = model.predict(x_test)
y_pred = y_predict.reshape(1,-1)
print(y_pred)



'''
loss :  [22.659046173095703, 3.3358426094055176]
[[43.77808   28.249557  14.215537  15.24366   29.169548  28.123674
  46.27001   12.826518  36.70836   10.438999  29.192589  15.022283
  18.520134  21.73857   20.852465  23.662563  10.808778  32.039936
  24.625856  23.730097  13.105374  19.013554  22.060669  29.82703
  32.20947   19.283552  27.063755  17.532398  34.140324  29.209297
  20.325493  19.227451  38.982445  40.15829   23.718046  21.374477
  13.157329  18.9534     9.686317  35.005615  20.541393  22.51985
  34.01038   14.02437   17.579954  23.185268  28.900763  16.15739
  24.910038  24.129538  36.522293  42.100487  19.123146  15.485992
  32.997818   9.406175  17.449152  16.299486  19.560854  20.371893
  29.8666    11.847276  35.0906    18.992483  11.342515  21.162817
  21.810564  19.781466  13.577249  19.694618  20.616722  23.189112
  18.443342  18.737932  26.053778  16.280647  46.91958   14.66596
  32.175648  14.102317  17.252064  19.396877  29.199959  14.899667
  14.200789  20.82148   21.611685  26.817013  21.208029  16.438852
  12.9964285 14.75274   32.746265  28.982656   9.72855   37.611572
  13.979218  31.997107  10.998709  20.869347  35.719475  19.093458 ]]
  '''


