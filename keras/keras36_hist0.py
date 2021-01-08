import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,LSTM

a = np.array(range(1,101))
size = 5                #size를 5로 정의하겠다.

def split_x(seq, size) :            #입력을 seq,size 받아들여서 아래와 같이 행동하겠다.
    aaa = []            # 빈 리스트 생성
    for i in range(len(seq)- size+1) :  # len(np.array(range(1,11)))은 리스트의 길이를
                                        # 의미하므로, 10이다. 10-5(size)+1 = 6
                                        # 즉, i값을 0부터6까지 아래 구문을 반복한다는 뜻.
        subset = seq[i :  (i+size)]     # subset= seq[i : (i+size)]
                                        # 해석하면 np.array(range(1,11))의 [i:(i+size)]
                                        # i부터 (i+size)-1까지의 리스트
        #aaa.append([item for item in subset])   # i가 반복될동안 subset에 있는 리스트를
                                                # 추가해내간다.
        aaa.append(subset)
    print(type(aaa))                    #aaa의 타입을 출력해라
    return np.array(aaa)                #aaa의 리스트를 출력해라

dataset = split_x(a,size)
x = dataset[:,:4]           # :(행),:(렬) => 슬라이싱
y = dataset[:,-1:]
print("====================")
print(dataset)
print(dataset.shape)            # (6,5)
print(x)
print(y)
x = x.reshape(x.shape[0],x.shape[1],1)
'''
aaa.append([item for item in subset])
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

aaa.append(subset)
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

 '''
# 2. model
model = load_model('./model/save_keras35.h5')
model.add(Dense(5,name='kingkeras1'))
model.add(Dense(1,name='kingkeras2'))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode = 'auto')

model.compile(loss = 'mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x,y,epochs=100, batch_size=32,verbose=1,validation_split=0.2,
 callbacks=[es])

print(hist)
hist1 = hist.history['loss']
print(np.array(hist1).shape)    #100,
print(hist.history.keys())
'''
dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
'''
#
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])       #회귀모델이기때문에 acc측정이 힘들다
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
plt.show()
