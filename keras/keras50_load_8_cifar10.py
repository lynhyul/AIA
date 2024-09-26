import numpy as np

x_train =np.load('../data/npy/cifar10_x_train.npy')
y_train = np.load('../data/npy/cifar10_y_train.npy')
x_test = np.load('../data/npy/cifar10_x_test.npy')
y_test = np.load('../data/npy/cifar10_y_test.npy')


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
# model.add(Conv2D(19,(2,2)))
# model.add(Dropout(0.2))
model.add(Conv2D(8,(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
for i in range(16) :
    model.add(Dense(300+i, activation= 'relu'))
    model.add(Dropout(0.2))
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

'''
for i in range(16) :
model.add(Conv2D(9,(2,2)))    
loss :  [1.671017050743103, 0.3765000104904175]

model.add(Conv2D(8,(2,2)))
loss :  [1.6304244995117188, 0.39660000801086426]

model.add(Dense(300+i*2, activation= 'relu'))
loss :  [1.7897374629974365, 0.3513999879360199]

model.add(Dense(200+i, activation= 'relu'))
loss :  [1.8676695823669434, 0.2996000051498413]

model.add(Dense(100+i, activation= 'relu'))
loss :  [1.7981146574020386, 0.29409998655319214]
'''