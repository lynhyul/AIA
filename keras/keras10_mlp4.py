#실습
# x는 (100, 5) 데이터 임의로 구성
# y는 (100, 2) 데이터 임의로 구성
# 모델을 완성하시오.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(100), range(301,401), range(1,101),  range(401,501),  range(501,601)])
y = np.array([range(711,811), range(1,101)])

print(x.shape)  # (5,100)
print(y.shape)  # (2,100 )


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
model.add(Dense(10,input_dim =5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))


#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train,y_train, epochs = 100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :',loss)
print('maae :',mae)

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
loss : 4.94338525669491e-09
maae : 4.998743679607287e-05
[[719.00006     9.00003  ]
 [804.0001     94.       ]
 [714.9999      5.000045 ]
 [715.9998      6.000042 ]
 [763.00006    52.999977 ]
 [752.         42.       ]
 [710.99994     1.0000845]
 [784.         73.999954 ]
 [799.         88.99996  ]
 [779.00006    69.00001  ]
 [735.99994    25.999947 ]
 [729.         19.000051 ]
 [736.9999     27.000004 ]
 [740.         30.000095 ]
 [777.         66.99996  ]
 [761.         51.000004 ]
 [790.9998     80.99989  ]
 [756.         46.00006  ]
 [748.99994    39.00004  ]
 [769.0001     59.00001  ]]
RMSE :  7.03092130117426e-05
R2 :  0.9999999999937451
'''