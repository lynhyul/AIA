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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters = 30, kernel_size =(4,4), padding = 'same', strides = 1, 
                                        input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(8,(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(315, activation= 'relu'))
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience= 5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=5, batch_size=32, validation_split=0.2,  
                                     callbacks = [early_stopping, cp])

#4. evaluate , predict

result = model.evaluate(x_test,y_test, batch_size=32)
print("loss : ", result[0])
print("accuracy : ", result[1])




import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"

plt.figure(figsize= (10,6)) # 판을 깔아준다.

plt.subplot(2, 1, 1)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 1번째   
plt.plot(hist.history['loss'], marker='.', c='red',label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue',label='val_loss')
plt.grid()

plt.title('손실비용')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 2번째   
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()

plt.title('정확도')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])


plt.show()

# y_predict = model.predict(x_test)
# y_Mpred = np.argmax(y_predict,axis=-1)
# print("y_test : ",y_test[:10])
# print("y_test : ",y_test[:10])


#완성하시오.
#지표는 acc /// 0.985 이상


'''
loss :  [0.05071277171373367, 0.9900000095367432]
'''

'''
# filters = 100
loss :  [0.06288700550794601, 0.9871000051498413]
loss :  [0.08189371973276138, 0.9847000241279602]
loss :  [0.08939924091100693, 0.9829999804496765]

#filters = 10 => 속도가 100에 비해 빨라지는것을 확인
loss :  [0.07050281018018723, 0.9861999750137329]

node (300->500) 증가
loss :  [0.07630899548530579, 0.98580002784729]
node (300 -> 200) 감소
loss :  [0.0724119246006012, 0.9850000143051147]
layer (1->2개) 증가
loss :  [0.07203719019889832, 0.9853000044822693]

#filters = 20
loss :  [0.06127446889877319, 0.9876999855041504]

#filters = 30
loss :  [0.06042682006955147, 0.9879999756813049]
kernel_size = c_layer1 (4,4)
loss :  [0.051556773483753204, 0.988099992275238]
loss :  [0.08526843786239624, 0.9886000156402588]
loss :  [0.060998767614364624, 0.9886000156402588]

#filters = 30 / layer2 = 8
loss :  [0.060668256133794785, 0.9889000058174133]
loss :  [0.05071277171373367, 0.9900000095367432]

kernel_size = c_layer1(4,4) , c_layer2(4,4)
loss :  [0.06441042572259903, 0.9887999892234802]

#filters = 31
loss :  [0.07249615341424942, 0.9855999946594238]

#filters = 40
loss :  [0.07983095943927765, 0.9868000149726868]

#filters = 50
loss :  [0.08281136304140091, 0.9861000180244446]

#hidden layer 삭제 => acc가 낮아지는것을 확인
loss :  [0.07945937663316727, 0.9749000072479248]


'''


#응용
# y_test 10개와 y_test 10개를 출력하시오
# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)






# plt.imshow(x_train[0], 'gray')
# plt.show()

