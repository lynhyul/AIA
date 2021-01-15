# Early Stopping의 단점을 개선해보자 ModelCheckPoint


import numpy as np

from tensorflow.keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000,28,28) (60000,)    => 60000,28,28,1
print(x_test.shape, y_test.shape)       # (10000,28,28) (10000,)    => 10000,28,28,1

print(x_train[0])   #    
print("y_train = ", y_train[0])   # 5

print(x_train[0].shape)     #(28, 28)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.         # 이미지의 전처리 (max값이 255기때문에 255로 나눠서
                                                                                            #0~1 사이로 만듦)
x_test = x_test.reshape(10000,28,28,1)/255.
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

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# model = Sequential()

# model.add(Conv2D(filters = 30, kernel_size =(4,4), padding = 'same', strides = 1, 
#                                         input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# for a in range(4) :
#     model.add(Conv2D(14-a*2,(2,2)))
#     model.add(Dropout(0.2))
# model.add(Flatten())
# for i in range(5) :
#     model.add(Dense(310+i, activation= 'relu'))
# model.add(Dense(10, activation='softmax'))



# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# early_stopping = EarlyStopping(monitor='val_loss', patience= 5)
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
# hist = model.fit(x_train,y_train, epochs=100, batch_size=120, validation_split=0.2,  
#                                      callbacks = [early_stopping, cp])

model = load_model('../data/h5/k51_1_model2.h5')

#4. evaluate , predict

result = model.evaluate(x_test,y_test, batch_size=32)
print("loss : ", result[0])
print("accuracy : ", result[1])

'''
313/313 [==============================] - 1s 2ms/step - loss: 0.0435 - acc: 0.9898
loss :  0.04350564256310463
accuracy :  0.989799976348877
'''


# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import matplotlib.font_manager as fm
# import matplotlib
# matplotlib.rcParams['axes.unicode_minus'] = False 
# matplotlib.rcParams['font.family'] = "Malgun Gothic"

# plt.figure(figsize= (10,6)) # 판을 깔아준다.

# plt.subplot(2, 1, 1)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 1번째   
# plt.plot(hist.history['loss'], marker='.', c='red',label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue',label='val_loss')
# plt.grid()

# plt.title('손실비용')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# plt.subplot(2, 1, 2)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 2번째   
# plt.plot(hist.history['acc'], marker='.', c='red')
# plt.plot(hist.history['val_acc'], marker='.', c='blue')
# plt.grid()

# plt.title('정확도')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc','val_acc'])


# plt.show()

# y_predict = model.predict(x_test)
# y_Mpred = np.argmax(y_predict,axis=-1)
# print("y_test : ",y_test[:10])
# print("y_test : ",y_test[:10])
