#실습
# x는 (100, 5) 데이터 임의로 구성
# y는 (100, 2) 데이터 임의로 구성
# 모델을 완성하시오.

# 다 만든 친구들은 predict의 일부값을 출력하시오.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
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

model = Sequential()
model.add(Dense(10,input_dim =5, activation = 'relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))


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

y_pred2 = model.predict(x_pred2)
print(y_pred2)


'''
loss : 1.5507645168000295e-09
maae : 2.7780235541285947e-05
[[719.           8.999998  ]
 [803.99994     93.99999   ]
 [715.           4.9999843 ]
 [716.           5.9999743 ]
 [763.00006     52.999973  ]
 [752.          42.00001   ]
 [711.           0.99998504]
 [784.00006     73.9999    ]
 [798.99994     89.00005   ]
 [778.99994     68.99998   ]
 [736.          26.000023  ]
 [729.          18.999998  ]
 [736.99994     27.00002   ]
 [740.          30.000013  ]
 [776.99994     66.99996   ]
 [761.00006     50.999924  ]
 [791.          81.        ]
 [755.99994     46.00003   ]
 [749.          39.000015  ]
 [768.99994     58.999996  ]]
RMSE :  3.937974839195155e-05
R2 :  0.9999999999980378

# 과제 수행 이후
loss : 2.4890325356352605e-09
mae : 3.0165911084623076e-05
[[719.00006     9.000017 ]
 [804.         94.00001  ]
 [715.00006     4.999977 ]
 [715.99994     5.9999957]
 [762.99994    53.000023 ]
 [752.         42.000008 ]
 [711.          0.9999993]
 [784.         73.999985 ]
 [799.00006    88.999954 ]
 [779.         68.99999  ]
 [735.9999     25.999996 ]
 [729.         18.999989 ]
 [736.99994    26.999989 ]
 [739.99994    30.000017 ]
 [777.         66.999985 ]
 [761.         50.999985 ]
 [791.         81.000015 ]
 [755.9998     45.999992 ]
 [749.00006    38.999966 ]
 [768.9999     58.999992 ]]
RMSE :  4.9890205696640525e-05
R2 :  0.9999999999968506
[[812.15906 100.89015]]
'''