#sklearn 데이터셋
#LSTM 으로 모델링
#Dense와 성능비교
#다중분류


import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf

#x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target
#print(dataset.DESCR)    
#print(dataset.feature_names)        # sepal(꽃받침), petal(꽃잎)
print(x.shape)      # (150,4)
print(y.shape)      # (150,)

print(x[:5])
print(y)

## 원핫인코딩
#from tensorflow.keras.utils import to_categorical # 케라스 2.0버전
#from keras.utils.np_utils import to_categorical -> 케라스 1.0버전(구버전)
#y = to_categorical(y)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one = OneHotEncoder()
y = y.reshape(-1,1)
one.fit(y)
y = one.transform(y).toarray()

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
print(y)
print(x.shape)  # (150,4)
print(y.shape)  # (150,3)


'''
x[:5] = 
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
 y =
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)
x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(x_train.shape)    # 120,4
print(x_test.shape)     # 30,4

x_train = x_train.reshape(120,4,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#2. modeling

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,LSTM

input1 = Input(shape=(4,1))
lstm = LSTM(25, activation= 'relu',return_sequences=True) (input1)
lstm = LSTM(30, activation= 'relu') (lstm)
dense1 = Dense(100, activation= 'relu') (lstm)
dense1 = Dense(55, activation= 'relu') (dense1)
dense1 = Dense(25, activation= 'relu') (dense1)
output1 = Dense(3, activation='softmax') (dense1)

model = Model(input1, output1)

#3. compile fit

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 5, mode = 'auto')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=300, batch_size=8, validation_data=(x_val, y_val),  
                                     callbacks = early_stopping)

#4. evaluate , predict

loss = model.evaluate(x_test,y_test, batch_size=1)
print("loss : ",loss)


y_predict = model.predict(x_test[0:6])
y_Mpred = np.argmax(y_predict,axis=-1)
print("y_predcit: ",y_predict)
print("y_MaxPredict: ",y_Mpred)
print(y_test[0:6])


#결과치 나오게 코딩할것 argmax

'''
loss :  [0.1539025902748108, 0.8999999761581421]
y_predcit:  [[7.55347873e-22 8.53142003e-04 9.99146819e-01]
 [2.60867528e-04 9.60692704e-01 3.90464216e-02]
 [9.95145261e-01 4.85303160e-03 1.64619303e-06]
 [2.15849286e-05 8.84831131e-01 1.15147196e-01]
 [9.95605409e-01 4.39408422e-03 4.18441914e-07]
 [4.47141845e-03 9.85268295e-01 1.02602364e-02]]
y_MaxPredict:  [2 1 0 1 0 1]
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
 '''