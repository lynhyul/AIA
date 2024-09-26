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


#2. modeling

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input,Dropout

model = Sequential()
model.add(Dense(1000, input_shape=(4,)))
model.add(Dense(500))
model.add(Dropout(0.2))
model.add(Dense(500))
model.add(Dropout(0.3))
model.add(Dense(500))
model.add(Dropout(0.4))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(3, activation= 'softmax'))  #다중분류에서는 가지고싶은 결과 수 만큼 입력한다.

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
적용 전
loss :  [0.21537503600120544, 0.9333333373069763]
y_predcit:  [[2.2971914e-08 9.1148734e-02 9.0885127e-01]
 [2.4477083e-03 9.1968471e-01 7.7867553e-02]
 [9.9985898e-01 1.4105017e-04 1.1314766e-08]
 [2.6272690e-02 9.2638808e-01 4.7339171e-02]
 [9.9999535e-01 4.6884725e-06 2.1943765e-10]
 [4.8054352e-02 9.3721974e-01 1.4725838e-02]]
y_MaxPredict:  [2 1 0 1 0 1]
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]

loss :  [0.09372270107269287, 0.9666666388511658]
y_predcit:  [[4.0978970e-32 1.7707961e-06 9.9999821e-01]
 [5.0439975e-11 8.5046697e-01 1.4953309e-01]
 [1.0000000e+00 1.4891668e-14 2.0197622e-23]
 [1.2269163e-06 9.2184359e-01 7.8155212e-02]
 [1.0000000e+00 3.0871362e-21 2.7510585e-30]
 [9.2283062e-07 9.9774480e-01 2.2543138e-03]]
y_MaxPredict:  [2 1 0 1 0 1]
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]


 '''