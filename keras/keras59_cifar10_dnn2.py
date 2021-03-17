# 다차원 댄스 모델
# (n,32,32,3) -> (n,10)

import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

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

# model.add(Conv2D(filters = 30, kernel_size =(4,4), padding = 'same', strides = 1, 
#                                         input_shape=(32,32,3)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu',input_shape=(32,32,3)))
# model.add(Conv2D(19,(2,2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(8,(2,2)))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(300, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './modelCheckpoint/k46_cifa10_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience= 5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist =model.fit(x_train,y_train, epochs=100, batch_size=120, validation_split=0.2,  
                                     callbacks = [early_stopping,cp])

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



import matplotlib.pyplot as plt

plt.figure(figsize= (10,6)) # 판을 깔아준다.

plt.subplot(2, 1, 1)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 1번째   
plt.plot(hist.history['loss'], marker='.', c='red',label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue',label='val_loss')
plt.grid()

plt.title('Cost loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 2번째   
plt.plot(hist.history['acc'], marker='.', c='red',label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue',label='val_acc')
plt.grid()

plt.title('Accuarcy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.show()