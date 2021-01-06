#과제 및 실습 LSTM
# 전처리, earlystopping등등 다넣어보자
# 데이터 1~100
#    x         y
# 1,2,3,4,5    6
#.....
# 95,96,97,98,99,100




import numpy as np

a = np.array(range(1,101))
b = np.array(range(96,106))
size = 6                #size를 5로 정의하겠다.

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
x_predict = split_x(b,size)
print(x_predict)

#predict를 만들것
#96,97,98,99,100 -> 101을 예측하는 모델
# .....
#100,101,102,103,104 -> 105
#예상 predict는 (101, 102,103,104,105)


x = dataset[:,:5]           # :(행),:(렬) => 슬라이싱
y = dataset[:,-1:]
x_pred = x_predict[:,:-1]

print(x_pred.shape) # (5,5)
print(x_pred)   

print(x.shape)  #95,5
print(y.shape)  #95,1

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2, 
                                            random_state=101)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)



print(x.shape)
print(x_train.shape)
print(x_pred.shape)


from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense,Input, LSTM

input1 = Input(shape=(5,))
dense1 = Dense(30, activation='relu') (input1)
dense1 = Dense(40, activation= 'relu') (dense1)
dense1 = Dense(50, activation= 'relu') (dense1)
dense1 = Dense(20, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1,output1)

model.compile(loss = 'mse', optimizer= 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor= 'loss', patience=10, mode='auto')
model.fit(x_train,y_train,epochs=200,batch_size=8,validation_data=(x_val,y_val),
            callbacks=early_stopping)

loss = model.evaluate(x_test,y_test)
print(loss)



x_pred1 = model.predict([x_pred])
x_pred1 = x_pred1.reshape(1,5)
print(x_pred1)


'''
[0.00023585744202136993, 0.00818950217217207]
[[101.00644  102.00662  103.006805 104.006996 105.00719 ]]
'''
