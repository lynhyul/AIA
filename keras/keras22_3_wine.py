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


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


print(x_train.shape)    # 113,13
print(x_test.shape)     # 36,13

#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input

model = Sequential()
model.add(Dense(30, activation= 'relu', input_shape = (13,) ))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))

#compile, fit
#from tensorflow.keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='loss', patience= 15, mode = 'auto')
#model.compile(loss = 'mean_squared_error', optimizer='adam', metrics =['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics =['accuracy','mae'])
model.fit(x_train,y_train, epochs = 100, batch_size=8, validation_data=(x_val,y_val)) 
                                #callbacks = early_stopping)

loss = model.evaluate(x_test,y_test, batch_size=1)
print("[loss, accuracy, mae] : ",loss)

y_predict = model.predict(x_test)
y_Mpred = np.argmax(y_predict,axis=-1)
print("y_predcit: ",y_predict)
print("y_MaxPredict: ",y_Mpred)
print("target = \n",y_test)

'''
[loss, accuracy, mae] :  [0.010274448432028294, 1.0, 0.00630617793649435]
y_predcit:  [[9.9999487e-01 5.1067154e-06 3.5782644e-08]
 [9.9970740e-01 2.9259940e-04 1.3819121e-08]
 [1.5855443e-07 4.0582331e-06 9.9999583e-01]
 [9.9999988e-01 9.5567664e-08 7.1128908e-10]
 [1.8731674e-08 1.0489656e-06 9.9999893e-01]
 [4.8449721e-08 9.9999988e-01 1.4247635e-07]
 [1.7589845e-07 2.7254068e-06 9.9999714e-01]
 [9.9999928e-01 6.6522563e-07 5.5616725e-09]
 [7.3107803e-05 9.9992692e-01 6.8793260e-09]
 [1.1146433e-06 9.9999893e-01 1.1643812e-08]
 [9.9999988e-01 8.3832013e-08 7.3045481e-10]
 [4.7152242e-04 9.9952817e-01 3.9639025e-07]
 [9.9025238e-01 9.7464323e-03 1.1777155e-06]
 [9.9997413e-01 2.5810114e-05 3.7669366e-08]
 [9.9999952e-01 4.9593956e-07 3.3705603e-09]
 [1.0000000e+00 5.1269833e-08 7.8755219e-10]
 [1.6247393e-08 1.0000000e+00 3.7455172e-09]
 [5.9431464e-08 8.4710591e-06 9.9999142e-01]
 [1.2094131e-09 1.0000000e+00 7.1301431e-10]
 [7.6742829e-10 1.0000000e+00 2.8649583e-10]
 [3.5324985e-10 1.0000000e+00 3.1862377e-10]
 [2.7484280e-06 8.5056312e-02 9.1494095e-01]
 [5.9946087e-06 3.7021674e-02 9.6297234e-01]
 [4.5566564e-09 2.3853610e-07 9.9999976e-01]
 [1.0000000e+00 4.6500958e-08 7.7397855e-10]
 [9.9999893e-01 1.1088894e-06 7.3232669e-09]
 [2.1881563e-09 1.0000000e+00 7.2722700e-10]
 [3.0748024e-09 1.0000000e+00 7.9328606e-09]
 [7.0284378e-09 1.1791766e-06 9.9999881e-01]
 [1.8881957e-08 1.0000000e+00 3.7601858e-08]
 [4.7448790e-08 1.5870469e-06 9.9999833e-01]
 [2.6196726e-06 9.9901783e-01 9.7951002e-04]
 [3.2816422e-06 7.9366624e-01 2.0633040e-01]
 [9.9999917e-01 7.8030541e-07 6.4898016e-09]
 [9.9950957e-01 4.9043860e-04 4.5252438e-08]
 [9.9999988e-01 6.7271934e-08 1.0747185e-09]]
y_MaxPredict:  [0 0 2 0 2 1 2 0 1 1 0 1 0 0 0 0 1 2 1 1 1 2 2 2 0 0 1 1 2 1 2 1 1 0 0 0]
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