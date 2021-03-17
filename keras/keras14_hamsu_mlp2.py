#다:다 mlp 함수형
#keras10_mlp3을 함수로 바꾸시오.


import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터
x = np.array([range(100), range(301,401), range(1,101)])
y = np.array([range(711,811), range(1,101), range(201,301)])

print(x.shape)  # (3,100)
print(y.shape)  # (3,100 )


x = np.transpose(x)
y = np.transpose(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 ,shuffle = True,\
    random_state=66)


print(x_train.shape)    # (80,3)
print(y_train.shape)    # (80,3)

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
Dense1 = Dense(10, activation='relu') (input1)
Dense2 = Dense(3) (Dense1)  # dense1의 출력을 입력으로 받아들인다.
Dense3 = Dense(4) (Dense2)  # dense2의 출력을 입력으로
Dense4 = Dense(5) (Dense3)
outputs = Dense(3) (Dense4) # dense3의 출력을 입력으로
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
loss : 0.38148266077041626
mae : 0.10121749341487885
[[718.99994     9.000008  208.99998  ]
 [803.99994    94.000015  294.00003  ]
 [714.9998      5.0000496 204.99995  ]
 [715.9998      6.000005  205.99995  ]
 [762.99994    53.000046  253.00005  ]
 [751.99994    42.00002   241.99994  ]
 [706.4077      2.3339367 200.85686  ]
 [783.99994    74.00001   273.99997  ]
 [798.99976    89.00001   288.99997  ]
 [778.9998     69.000046  268.99997  ]
 [735.99976    26.000044  225.99997  ]
 [728.9999     19.000015  218.99994  ]
 [736.9998     26.999987  226.99995  ]
 [739.99994    30.00001   230.       ]
 [776.9999     67.00001   266.99997  ]
 [760.9999     51.000023  250.99997  ]
 [790.9998     81.000015  280.99997  ]
 [755.99994    46.000027  245.99998  ]
 [748.9998     39.000076  238.99997  ]
 [768.9998     59.000046  258.99997  ]]
RMSE :  0.6176428357729203
R2 :  0.9995173106432693
'''