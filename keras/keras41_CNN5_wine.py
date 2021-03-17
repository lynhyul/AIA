#CNN으로 구성
# 2차원을 4차원으로 늘려서 사용하시오.


#CNN으로 구성
# 2차원을 4차원으로 늘려서 사용하시오.

import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()

print(dataset.DESCR)
print(dataset.feature_names)    # class = 3

x = dataset.data
y = dataset.target

print(x.shape)  # 178,13
print(y.shape)  # 178

print(x)
print(y)
'''
x =
[[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]
 [1.320e+01 1.780e+00 2.140e+00 ... 1.050e+00 3.400e+00 1.050e+03]
 [1.316e+01 2.360e+00 2.670e+00 ... 1.030e+00 3.170e+00 1.185e+03]
 ...
 [1.327e+01 4.280e+00 2.260e+00 ... 5.900e-01 1.560e+00 8.350e+02]
 [1.317e+01 2.590e+00 2.370e+00 ... 6.000e-01 1.620e+00 8.400e+02]
 [1.413e+01 4.100e+00 2.740e+00 ... 6.100e-01 1.600e+00 5.600e+02]]
 y =
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
 '''

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one = OneHotEncoder()
y = y.reshape(-1,1)
one.fit(y)
y = one.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8, 
                                                    random_state=101)

# x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 0.8)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)


print(x_train.shape)    # 142,13
print(x_test.shape)     # 36,13

x_train = x_train.reshape(x_train.shape[0],13,1,1)
x_test = x_test.reshape(x_test.shape[0],13,1,1)

#model

#2. modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


model = Sequential()

model.add(Conv2D(filters = 30, kernel_size =(2,2), padding = 'same', strides = 1, 
                                        input_shape=(13,1,1)))

model.add(Flatten())
model.add(Dense(315, activation= 'relu'))
model.add(Dense(315, activation= 'relu'))
model.add(Dense(50))
model.add(Dense(3, activation= 'softmax'))  #다중분류에서는 가지고싶은 결과 수 만큼 입력한다.

#compile, fit
#from tensorflow.keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='loss', patience= 15, mode = 'auto')
#model.compile(loss = 'mean_squared_error', optimizer='adam', metrics =['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics =['accuracy','mae'])
model.fit(x_train,y_train, epochs = 100, batch_size=8) 
                                #callbacks = early_stopping)

loss = model.evaluate(x_test,y_test, batch_size=1)
print("[loss, accuracy, mae] : ",loss)

y_predict = model.predict(x_test)
y_Mpred = np.argmax(y_predict,axis=-1)
print("y_predcit: ",y_predict)
print("y_MaxPredict: ",y_Mpred)
print("target = \n",y_test)



'''
[loss, accuracy, mae] :  [0.1538127064704895, 0.9722222089767456, 0.0513802096247673]
'''
