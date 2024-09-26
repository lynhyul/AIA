#
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

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 0.8)


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)


print(x_train.shape)    # 113,13
print(x_test.shape)     # 36,13


x_trian = x_train.reshape(113,13,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input,LSTM

input1 = Input(shape = (13,1))
lstm = LSTM(400, activation= 'relu',return_sequences=False) (input1)
dense1 = Dense(150, activation='relu') (lstm)
dense1 = Dense(80, activation= 'relu') (dense1)
dense1 = Dense(50, activation= 'relu') (dense1)
dense1 = Dense(60, activation= 'relu') (dense1)
output1 = Dense(3, activation= 'softmax') (dense1)

model = Model(input1,output1)

#compile, fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 10, mode = 'auto')
#model.compile(loss = 'mean_squared_error', optimizer='adam', metrics =['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics =['accuracy','mae'])
model.fit(x_train,y_train, epochs = 100, batch_size=8, validation_data=(x_val,y_val),
                                callbacks = early_stopping)

loss = model.evaluate(x_test,y_test, batch_size=1)
print("[loss, accuracy, mae] : ",loss)


y_predict = model.predict(x_test)
y_Mpred = np.argmax(y_predict,axis=-1)
print("y_predcit: ",y_predict)
print("y_MaxPredict: ",y_Mpred)
print("target = \n",y_test)

'''
[loss, accuracy, mae] :  [0.7084261178970337, 0.9722222089767456, 0.022283997386693954]
y_predcit:  [[1.00000000e+00 8.78104230e-13 3.30229009e-19]
 [1.00000000e+00 3.81107645e-11 3.72731488e-17]
 [2.37785347e-10 1.02534170e-12 1.00000000e+00]
 [1.00000000e+00 1.33434144e-13 1.97093892e-20]
 [7.35920047e-10 3.40099260e-12 1.00000000e+00]
 [3.06543163e-25 1.00000000e+00 0.00000000e+00]
 [1.63585673e-11 2.20995690e-14 1.00000000e+00]
 [1.00000000e+00 1.04951869e-16 5.99402154e-25]
 [7.78592420e-17 1.00000000e+00 3.37375890e-35]
 [2.74794595e-03 9.97252047e-01 1.98535077e-11]
 [1.00000000e+00 1.01825184e-19 4.79359329e-29]
 [1.23485513e-01 8.76514435e-01 2.36755349e-09]
 [9.93853927e-01 6.14604121e-03 6.51449517e-09]
 [1.00000000e+00 1.70363879e-11 2.05213849e-17]
 [9.99999762e-01 2.80889225e-07 2.59506409e-11]
 [1.00000000e+00 1.29133324e-17 6.28560795e-26]
 [1.59686363e-28 1.00000000e+00 0.00000000e+00]
 [3.06159009e-08 6.53980536e-10 1.00000000e+00]
 [2.33749215e-31 1.00000000e+00 0.00000000e+00]
 [1.09059850e-31 1.00000000e+00 0.00000000e+00]
 [5.20329573e-22 1.00000000e+00 3.45074427e-38]
 [6.52763310e-09 1.00000000e+00 1.04017081e-11]
 [3.41518791e-09 1.34423958e-10 1.00000000e+00]
 [8.47241114e-13 2.73781351e-16 1.00000000e+00]
 [1.00000000e+00 5.58282857e-16 1.99781580e-23]
 [1.00000000e+00 1.93478024e-12 2.49955659e-18]
 [5.91937038e-20 1.00000000e+00 1.12660709e-35]
 [5.41927180e-13 1.00000000e+00 8.94184691e-23]
 [7.03792421e-06 1.56180988e-07 9.99992847e-01]
 [2.21797749e-24 1.00000000e+00 0.00000000e+00]
 [7.13793608e-11 7.61586703e-14 1.00000000e+00]
 [2.29315081e-20 1.00000000e+00 4.21725312e-34]
 [3.95182008e-03 9.29087579e-01 6.69606477e-02]
 [1.00000000e+00 1.37858789e-12 6.59764582e-19]
 [9.99963045e-01 3.69884001e-05 8.97890151e-10]
 [1.00000000e+00 2.67022293e-16 1.16781651e-23]]
y_MaxPredict:  [0 0 2 0 2 1 2 0 1 1 0 1 0 0 0 0 1 2 1 1 1 1 2 2 0 0 1 1 2 1 2 1 1 0 0 0]
target =
 [[1. 0. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]]
 '''