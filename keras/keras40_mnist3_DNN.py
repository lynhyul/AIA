# 주말과제
# dense 모델로 구성 input_shape = (28*28, )

# 인공지능계의 hello world라 불리는 mnist!!!

import numpy as np
import matplotlib.pyplot as plt

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Dense(310, activation='relu', input_shape=(28*28,)))
model.add(Dense(315, activation= 'relu'))
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


#완성하시오.
#지표는 acc /// 0.985 이상




#응용
# y_test 10개와 y_test 10개를 출력하시오
# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)






# plt.imshow(x_train[0], 'gray')
# plt.show()

