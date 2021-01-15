#acc = 0.3

import numpy as np

a = np.array(range(1,11))
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
    # print(type(aaa))                    #aaa의 타입을 출력해라
    return np.array(aaa)                #aaa의 리스트를 출력해라

dataset = split_x(a,size)
# print("====================")
# print(dataset)
# print(dataset.shape)            # (6,5)

x = dataset[:,:4]           # :(행),:(렬) => 슬라이싱
y = dataset[:,-1:]

print(x.shape)
print(y.shape)


from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense,Input, LSTM

input1 = Input(shape=(4,))
dense1 = Dense(30, activation='relu') (input1)
dense1 = Dense(40, activation= 'relu') (dense1)
dense1 = Dense(50, activation= 'relu') (dense1)
dense1 = Dense(20, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1,output1)

model.compile(loss = 'mse', optimizer= 'adam', metrics = ['mae'])
model.fit(x,y,epochs=300,batch_size=8)

loss = model.evaluate(x,y)
print(loss)

y_predict = dataset[-1:,1:]
y_predict = y_predict.reshape(1,-1,1)
y_predict = model.predict(y_predict)

print(y_predict)

'''
[0.00016123533714562654, 0.010580222122371197]
[[11.093985]]
'''