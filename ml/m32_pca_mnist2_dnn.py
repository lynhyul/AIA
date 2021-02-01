# m31로 만든 1.0 이상의 n_component= 1를 사용하여
# dnn모델을 만들어보자

# mnist dnn보다 성능을 좋게 만들어봐라
# cnn과 비교!!


# 주말과제
# dense 모델로 구성 input_shape = (28*28, )

# 인공지능계의 hello world라 불리는 mnist!!!

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tensorflow.keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000,28,28) (60000,)    => 60000,28,28,1
print(x_test.shape, y_test.shape)       # (10000,28,28) (10000,)    => 10000,28,28,1

print(x_train[0])   #    
print("y_train = ", y_train[0])   # 5

print(x_train[0].shape)     #(28, 28)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.         # 이미지의 전처리 (max값이 255기때문에 255로 나눠서
                                                                                            #0~1 사이로 만듦)
x_test = x_test.reshape(10000,28*28)/255.
#(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1) ) # x_test = x_train.reshape(10000,28,28,1)/255.

#OneHotEncoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
print(y_train.shape)            # (60000,10)
print(y_test.shape)            # (10000,10)

pca = PCA(n_components=1)
x_train = pca.fit_transform(x_train)  # merge fit,transform
x_test = pca.transform(x_test)

print(x_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(1,)))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 5, mode = 'auto')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=100, batch_size=120, validation_split=0.2,  
                                     callbacks = early_stopping)

#4. evaluate , predict

loss = model.evaluate(x_test,y_test, batch_size=1)
print("loss : ",loss)


y_predict = model.predict(x_test)
y_Mpred = np.argmax(y_predict,axis=-1)
print("y_test : ",y_test[:10])
print("y_test : ",y_test[:10])


'''
기존의 DNN : loss :  [0.9793000221252441]
PCA적용 후의 DNN :  loss :  [0.9789000153541565]
'''