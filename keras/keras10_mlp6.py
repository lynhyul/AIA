#실습
#mlp4처럼 predict값을 도출 할 것
#1:다 mlp


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(100)])
y = np.array([range(711,811), range(1,101), range(201,301)])

print(x.shape)  # (5,100)
print(y.shape)  # (2,100 )
x_pred2 = np.array([100])

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
model.add(Dense(100,input_dim =1, activation = 'relu'))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(3))


#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train,y_train, epochs = 200, batch_size=2, validation_split=0.3)

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
loss : 7.032349094515666e-05
mae : 0.005260380916297436
[[719.02155     8.999599  209.00764  ]
 [803.99493    93.99966   293.9974   ]
 [715.0229      4.9996    205.0081   ]
 [716.0225      5.999594  206.00801  ]
 [763.0078     52.999645  253.00232  ]
 [752.01117    41.999622  242.00365  ]
 [711.0241      0.9995999 201.00862  ]
 [784.00116    73.99964   273.99982  ]
 [798.9966     88.99966   288.99802  ]
 [779.00275    68.999626  269.0004   ]
 [736.0163     25.999619  226.00558  ]
 [729.01843    18.999607  219.00644  ]
 [737.0159     26.999619  227.00546  ]
 [740.01495    29.999617  230.0051   ]
 [777.00336    66.99964   267.00064  ]
 [761.0084     50.99963   251.00256  ]
 [790.999      80.99965   280.99896  ]
 [756.01       45.999622  246.0032   ]
 [749.0122     38.999626  239.00404  ]
 [769.0059     58.99964   259.00162  ]]
RMSE :  0.008385910389161937
R2 :  0.9999999110198076
[[810.9928   100.999664 300.99655 ]]
'''