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

print(x.shape)  # 6,4
print(y.shape)  # 6,1

# from sklearn.model_selection import train_test_split
# x_train,x_test, y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=101)
# x_train,x_val, y_train,y_val = train_test_split(x_train,y_train,train_size=0.8)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)
x = x.reshape(x.shape[0],x.shape[1],1)
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

# print(x_train.shape)
# print(x_test.shape)



from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')

model.summary()

model.compile(loss = 'mse', optimizer= 'adam', metrics = ['mae'])
model.fit(x,y,epochs=300,batch_size=5,)

loss = model.evaluate(x,y)
print(loss)

y_predict = dataset[-1:,1:]
y_predict = y_predict.reshape(1,-1,1)
y_predict = model.predict(y_predict)

print(y_predict)

'''
[0.007834351621568203, 0.0801338329911232]
[[10.806564]]

WARNING:tensorflow:No training configuration found in the save file, 
so the model was *not* compiled. Compile it manually.
'''