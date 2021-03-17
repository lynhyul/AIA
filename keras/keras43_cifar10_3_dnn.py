import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

print(x_train.shape)    #50000,32,32,3
print(y_train.shape)    #50000,1
print(x_test.shape)     #10000,32,32,3

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[1]*3)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[1]*3)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
print(y_train.shape)            # (50000,10)
print(y_test.shape)            # (10000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

# model.add(Conv2D(filters = 30, kernel_size =(4,4), padding = 'same', strides = 1, 
#                                         input_shape=(32,32,3)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(Conv2D(9,(2,2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
model.add(Dense(150, activation='relu',input_shape = (32*32*3,)))
model.add(Dense(315, activation= 'relu'))
model.add(Dense(320, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 6, mode = 'auto')

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