import numpy as np

x_train =np.load('../data/npy/cifar100_x_train.npy')
y_train = np.load('../data/npy/cifar100_y_train.npy')
x_test = np.load('../data/npy/cifar100_x_test.npy')
y_test = np.load('../data/npy/cifar100_y_test.npy')


print(x_train.shape)    #50000,32,32,3
print(y_train.shape)    #50000,1
print(x_test.shape)     #10000,32,32,3

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

model.add(Conv2D(filters = 30, kernel_size =(4,4), padding = 'same', strides = 1, 
                                        input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
for a in range(4) :
    model.add(Conv2D(14-a*2,(2,2)))
    model.add(Dropout(0.2))
model.add(Flatten())
for i in range(5) :
    model.add(Dense(310+i, activation= 'relu'))
model.add(Dense(100, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 4, mode = 'auto')

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
