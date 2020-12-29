import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

model = Sequential()
model.add(Dense(10,input_dim =3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))


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
loss : 3.277571522630751e-06
maae : 0.0013907949905842543
[[719.0008      9.002897  208.9962   ]
 [803.999      93.99776   294.00305  ]
 [715.0009      5.0031657 204.99591  ]
 [716.00085     6.003082  205.99599  ]
 [762.9999     53.000237  252.99974  ]
 [752.0001     42.000927  241.9989   ]
 [711.0009      1.0034316 200.99559  ]
 [783.9993     73.99903   274.00146  ]
 [798.9991     88.99811   289.00266  ]
 [778.9995     68.99929   269.00107  ]
 [736.00055    26.001842  225.99759  ]
 [729.0006     19.00226   218.997    ]
 [737.00037    27.00185   226.9977   ]
 [740.00037    30.001667  229.99792  ]
 [776.9995     66.99945   267.0009   ]
 [760.9998     51.0004    250.9996   ]
 [790.99927    80.99864   281.00208  ]
 [756.00006    46.00066   245.99918  ]
 [749.0002     39.00111   238.99866  ]
 [768.9997     58.999866  259.0002   ]]
RMSE :  0.0018104064337507307
R2 :  0.9999999958528946
'''