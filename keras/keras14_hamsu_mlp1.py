#다:1 mlp 함수형
#keras10_mlp2 함수로 바꾸시오.

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

x = np.array([range(100), range(301,401), range(1,101)])
y = np.array(range(711,811))

print(x.shape)  # (3,100)
print(y.shape)  # (100,)

x = np.transpose(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 ,shuffle = True,\
    random_state=66)

#x_train, x_test, y_train, y_test = train_test_split(x,y, \
                           # train_size = 0.7, test_size =0.2,shuffle = True)

print(x_train.shape)  # (80,3)
print(y_train.shape)  # (80,)
'''
(3, 100)
(100,)
(80, 3)
(80,)
'''


#2. 모델 구성

# from keras.layers import Dense 위와 똑같이 불러오나, keras1에선 위 보다 불러오는게 조금 느려진다.

'''
model = Sequential()
model.add(Dense(10,input_dim =3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))
'''

input1 = Input(shape=(3,))
aaa = Dense(10, activation='relu') (input1)
aaa = Dense(3) (aaa)  # dense1의 출력을 입력으로 받아들인다.
aaa = Dense(4) (aaa)  # dense2의 출력을 입력으로
outputs = Dense(1) (aaa) # dense3의 출력을 입력으로
model = Model(input1, outputs)  #입출력 정확하게 명시

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train,y_train, epochs = 100, batch_size=1, validation_split=0.2)

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

'''
loss : 2.7939677238464355e-09
mae : 3.356933666509576e-05
[[718.9999 ]
 [804.     ]
 [714.9999 ]
 [715.99994]
 [763.00006]
 [752.     ]
 [711.     ]
 [784.00006]
 [799.     ]
 [779.     ]
 [736.     ]
 [728.99994]
 [736.99994]
 [740.     ]
 [777.     ]
 [761.     ]
 [791.     ]
 [755.99994]
 [749.     ]
 [768.99994]]
RMSE :  5.285799583645255e-05
R2 :  0.9999999999964648
'''