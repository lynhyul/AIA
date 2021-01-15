#1:다 mlp 함수형
#keras10_mlp6을 함수로 바꾸시오.



import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

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

'''
model = Sequential()
model.add(Dense(10,input_dim =1, activation = 'relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))
'''

input1 = Input(shape=(1,))
Dense1 = Dense(50, activation = 'relu') (input1)
Dense2 = Dense(10) (Dense1)
Dense3 = Dense(80) (Dense2)
Dense4 = Dense(15) (Dense3)
outputs = Dense(3) (Dense4)
model = Model(input1, outputs)

#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train,y_train, epochs = 100, batch_size=1, validation_split=0.3)

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
loss : 0.8702549934387207
mae : 0.5798295140266418
[[719.3651       8.990639   209.10167   ]
 [806.73364     93.94595    294.72617   ]
 [715.25366      4.992722   205.07227   ]
 [716.2815       5.992203   206.07962   ]
 [764.5912      52.967503   253.42496   ]
 [753.2846      41.973255   242.34412   ]
 [711.1422       0.99482006 201.0429    ]
 [786.17633     73.956436   274.57925   ]
 [801.5943      88.94858    289.68948   ]
 [781.037       68.959076   269.5425    ]
 [736.8388      25.981678   226.2266    ]
 [729.64374     18.985344   219.17513   ]
 [737.86664     26.981152   227.23393   ]
 [740.9503      29.979568   230.25595   ]
 [778.98126     66.96011    267.5278    ]
 [762.5354      50.968533   251.41028   ]
 [793.37134     80.95276    281.63065   ]
 [757.3961      45.971165   246.37352   ]
 [750.20105     38.974857   239.32208   ]
 [770.75836     58.964333   259.46902   ]]
RMSE :  0.9328745907066104
R2 :  0.998898867871881
[[813.92865 100.94225 301.77762]]
'''