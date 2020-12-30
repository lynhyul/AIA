import numpy as np

#1. 데이터
from tensorflow.keras.models import Sequential, Model   #함수형 모델(컨트롤 + 스페이스)
from tensorflow.keras.layers import Dense, Input

 
x = np.array([range(100), range(301,401), range(1,101),  range(100),  range(301,401)])
y = np.array([range(711,811), range(1,101)])

print(x.shape)  # (5,100)
print(y.shape)  # (2,100 )
x_pred2 = np.array([100,402,101,100, 401])
x_pred2 = x_pred2.reshape(1, 5)

print("x_pred2.shape : ", x_pred2.shape)    # x_pred2.shape = (1,5)

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 ,shuffle = True,\
     random_state=66)

print(x_train.shape)    # (80,3)
print(y_train.shape)    # (80,3)

#2. 모델 구성

# from keras.layers import Dense 위와 똑같이 불러오나, keras1에선 위 보다 불러오는게 조금 느려진다.

# model = Sequential()
# model.add(Dense(5,input_dim=1, activation='relu'))  # 컬럼의 갯수가 5개  
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(1))


input1 = Input(shape=(5,))
aaa = Dense(10, activation='relu') (input1)
aaa = Dense(3) (aaa)  # dense1의 출력을 입력으로 받아들인다.
aaa = Dense(4) (aaa)  # dense2의 출력을 입력으로
outputs = Dense(2) (aaa) # dense3의 출력을 입력으로
model = Model(input1, outputs)  #입출력 정확하게 명시

model.summary()


#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train,y_train, epochs = 100, batch_size=1, validation_split=0.2,
                        verbose =0)         # verbose = 0 => 시뮬레이션 결과를 보지 않겠다.

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :',loss)
print('mae :',mae)

y_predict = model.predict(x_test)
print(y_predict)


from sklearn.metrics import mean_squared_error      #mse
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))       #sqrt -> 루트를 씌우다
print("RMSE : ", RMSE (y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)

y_pred2 = model.predict(x_pred2)
print(y_pred2)


'''

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________

함수형
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 1)]               0
_________________________________________________________________
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10
=================================================================


_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
dense (Dense)                (None, 5)                 30
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10
=================================================================
Total params: 74
Trainable params: 74
Non-trainable params: 0
'''