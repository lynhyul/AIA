#다중분류

import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf
from sklearn.svm import LinearSVC

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
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# one = OneHotEncoder()
# y = y.reshape(-1,1)
# one.fit(y)
# y = one.transform(y).toarray()
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

print(y)
print(x.shape)  # (150,4)
print(y.shape)  # (150,3)


# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
#                                                     random_state=110)
# x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)


#2. modeling

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(10, input_shape=(4,)))
# model.add(Dense(5))
# model.add(Dense(3, activation= 'softmax'))  #다중분류에서는 가지고싶은 결과 수 만큼 입력한다.

model = LinearSVC()


#3. compile fit

# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience= 5, mode = 'auto')

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x,y)

#4. evaluate , predict

result = model.score(x,y)
print("result : ",result)


y_predict = model.predict(x[0:6])
y_Mpred = np.argmax(y_predict,axis=-1)
print("y_predcit: ",y_predict)
print("y_MaxPredict: ",y_Mpred)
print(y[0:6])


#결과치 나오게 코딩할것 argmax

'''
Deep learning
loss :  [0.01415738184005022, 1.0]
y_predcit:  [[1.7606923e-17 1.7337496e-07 9.9999988e-01]
 [4.2821070e-05 9.9479985e-01 5.1573813e-03]
 [9.9820244e-01 1.7975861e-03 2.6805536e-17]
 [2.8611670e-04 9.9705952e-01 2.6543736e-03]
 [9.9985671e-01 1.4330167e-04 4.4705715e-19]
 [9.3903748e-04 9.9903381e-01 2.7180373e-05]]
y_MaxPredict:  [2 1 0 1 0 1]      => 2 = (0~2)세 번째 인덱스가 가장 큰값 / 1 = 두 번째 인덱스가 가장
                                      큰 값
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]


Machine Learning
result :  0.9666666666666667
y_predcit:  [0 0 0 0 0 0]
y_MaxPredict:  0
[0 0 0 0 0 0]

 '''