#유방암 예측 모델을 만들어보자
#softmax를 적용한 이진분류 모델을 완성해보자.


import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(datasets.DESCR)
print(datasets.feature_names)
print(x.shape)      # 569,30
print(y.shape)      # 569,
print(x[:5])

from tensorflow.keras.utils import to_categorical # 케라스 2.0버전
#from keras.utils.np_utils import to_categorical -> 케라스 1.0버전(구버전)
y = to_categorical(y)

print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 110)




#전처리는 알아서 할 것

#2. modeling
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, activation= 'relu', input_shape = (30,) ))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(2, activation= 'softmax'))

#compile, fit
#from tensorflow.keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='loss', patience= 15, mode = 'auto')
#model.compile(loss = 'mean_squared_error', optimizer='adam', metrics =['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics =['accuracy','mae'])
model.fit(x_train,y_train, epochs = 100, batch_size=10, validation_split = 0.3) 
                                #callbacks = early_stopping)

loss = model.evaluate(x_test,y_test, batch_size=1)
print("[loss, accuracy, mae] : ",loss)

y_predict = model.predict(x_test[0:15])
y_Mpred = np.argmax(y_predict,axis=-1)
print("y_predcit: ",y_predict)
print("y_MaxPredict: ",y_Mpred)
print("target = ",y_test[0:15])

'''
[loss, accuracy, mae] :  [0.17527081072330475, 0.9385964870452881, 0.0907108336687088]
y_predcit:  [[6.1836294e-03 9.9381638e-01]
 [6.2763073e-02 9.3723696e-01]
 [2.6282005e-04 9.9973720e-01]
 [9.9956757e-01 4.3246779e-04]
 [2.9365592e-02 9.7063440e-01]
 [1.5630312e-01 8.4369689e-01]
 [2.0913752e-03 9.9790871e-01]
 [8.6325482e-03 9.9136740e-01]
 [1.6345013e-02 9.8365504e-01]
 [2.1751712e-01 7.8248292e-01]
 [8.0307603e-02 9.1969240e-01]
 [2.3436521e-03 9.9765635e-01]
 [1.0561855e-02 9.8943818e-01]
 [9.1920197e-01 8.0797948e-02]
 [4.7528666e-02 9.5247132e-01]]
y_MaxPredict:  [1 1 1 0 1 1 1 1 1 1 1 1 1 0 1]
target =  [[0. 1.]
 [0. 1.]
 [0. 1.]
 [1. 0.]
 [0. 1.]
 [1. 0.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [0. 1.]
 [1. 0.]
 [0. 1.]]
 '''