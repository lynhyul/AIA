#실습 train과 test 분리해서 소스를 완성하시오

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

print(x_train.shape)
print(y_train.shape)  # (100,3)
'''
(3, 100)
(100,)
(80, 3)
(80,)
'''


#2. 모델 구성

# from keras.layers import Dense 위와 똑같이 불러오나, keras1에선 위 보다 불러오는게 조금 느려진다.

model = Sequential()
model.add(Dense(10,input_dim =3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))


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
loss : 1.6763805898989403e-09
maae : 2.746581958490424e-05
[[737.     ]
 [762.     ]
 [724.     ]
 [735.     ]
 [753.00006]
 [725.     ]
 [759.     ]
 [738.     ]
 [809.     ]
 [784.00006]
 [773.00006]
 [715.00006]
 [759.99994]
 [728.     ]
 [717.00006]
 [791.     ]
 [774.00006]
 [794.99994]
 [785.00006]
 [751.     ]]
RMSE :  4.0943627517696345e-05
R2 :  0.9999999999977625
'''
 