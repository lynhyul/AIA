import numpy as np
#1. 데이터
x1 = np.array([range(100), range(301,401), range(1,101)])

y1 = np.array([range(711,811), range(1,101), range(201,301)])
y2 = np.array([range(501,601),range(711,811), range(100)])
# 결과 (3,100)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
# 결과 (100,3)


#x1,y1에대한 train_test_split
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test,y2_train, y2_test = train_test_split(x1,y1,y2, test_size = 0.2, shuffle = True,
                                            random_state=66)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model   
#앞 경로가 중복되는것은 위 처럼 ,로 같이 쓸 수 있다.
from tensorflow.keras.layers import Dense, Input

#모델 1
input1 = Input(shape=(3,))
dense1 = Dense(100, activation= 'relu') (input1)
dense1 = Dense(50, activation= 'relu') (dense1)
dense1 = Dense(50, activation= 'relu') (dense1)

#모델 분기1
output1 = Dense(30, activation= 'relu') (dense1)
output1 = Dense(70) (output1)
output1 = Dense(100) (output1)
output1 = Dense(3) (output1)

#모델 분기2
output2 = Dense(30, activation= 'relu') (dense1)
output2 = Dense(70) (output2)
output2 = Dense(70) (output2)
output2 = Dense(3) (output2)

#모델 선언
model = Model(inputs=input1,outputs=[output1,output2]) # 2개이상은 리스트로 취해서 넣는다.

model.summary()


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x1_train, [y1_train,y2_train], epochs= 300, batch_size=3,
                    validation_split= 0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate(x1_test,[y1_test,y2_test], batch_size=1)
print("loss : ", loss)


#loss :  [233.51632690429688, 97.42386627197266, 136.09246826171875, 97.42386627197266, 136.09246826171875]
#model.metrics_name :  ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mse', 'dense_15_mse']


y1_predict, y2_predict =model.predict([x1_test])

print("====================")
print("y1_predict : \n", y1_predict)
print("====================")
print("y2_predict : \n",y2_predict)


 #RMSE, R2

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict) :
        return np.sqrt(mean_squared_error(y_test,y_predict,))
RMSE1 = RMSE(y1_test,y1_predict)
RMSE2 = RMSE(y2_test,y2_predict)

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE(avg) : ", (RMSE2+RMSE1)/2)

R2_1 = r2_score(y1_test, y1_predict)
R2_2 = r2_score(y2_test, y2_predict)
print("R2_1 : ", R2_1)
print("R2_2 : ", R2_2)
print("R2(avg) : ", (R2_1+R2_2)/2)

'''
loss :  [0.09546250104904175, 0.020466584712266922, 0.07499592751264572, 0.020466584712266922, 0.07499592751264572]
====================
y1_predict :
 [[718.9231      9.049755  209.0212   ]
 [804.0876     94.11728   294.08157  ]
 [714.91754     5.047722  205.0193   ]
 [715.9189      6.048256  206.01976  ]
 [762.9848     53.072014  253.04211  ]
 [751.9694     42.06644   242.03688  ]
 [710.26764     1.5410327 201.136    ]
 [784.04095    74.1363    274.1041   ]
 [799.1267     89.14543   289.1168   ]
 [779.00726    69.08008   269.0497   ]
 [735.94696    26.058355  226.02927  ]
 [728.9371     19.054813  219.02594  ]
 [736.9483     27.058884  227.02975  ]
 [739.95264    30.060368  230.03117  ]
 [777.00446    67.07908   267.04868  ]
 [760.982      51.070984  251.0411   ]
 [791.1264     81.28049   281.24448  ]
 [755.97504    46.06848   246.03877  ]
 [748.96515    39.064922  239.03545  ]
 [768.9932     59.075035  259.0449   ]]
====================
y2_predict :
 [[5.0884268e+02 7.1879175e+02 8.0661850e+00]
 [5.9374872e+02 8.0366644e+02 9.3114616e+01]
 [5.0484842e+02 7.1479999e+02 4.0579515e+00]
 [5.0584702e+02 7.1579797e+02 5.0599914e+00]
 [5.5277917e+02 7.6270026e+02 5.2156891e+01]
 [5.4179504e+02 7.5172308e+02 4.1134228e+01]
 [5.0050980e+02 7.1003802e+02 5.1090151e-01]
 [5.7373633e+02 7.8362775e+02 7.3241531e+01]
 [5.8873206e+02 7.9862762e+02 8.8188095e+01]
 [5.6875604e+02 7.7866693e+02 6.8189888e+01]
 [5.2581812e+02 7.3575641e+02 2.5101233e+01]
 [5.1882825e+02 7.2877094e+02 1.8086794e+01]
 [5.2681671e+02 7.3675433e+02 2.6103333e+01]
 [5.2981232e+02 7.3974805e+02 2.9109472e+01]
 [5.6675891e+02 7.7667108e+02 6.6185738e+01]
 [5.5078204e+02 7.6070435e+02 5.0152767e+01]
 [5.8069379e+02 7.9053760e+02 8.0362907e+01]
 [5.4578931e+02 7.5571484e+02 4.5142452e+01]
 [5.3879938e+02 7.4872937e+02 3.8128025e+01]
 [5.5877045e+02 7.6868768e+02 5.8169262e+01]]
RMSE1 :  0.1430565421644625
RMSE2 :  0.2738679320862377
RMSE(avg) :  0.2084622371253501
R2_1 :  0.9999741054508973
R2_2 :  0.9999050980204723
R2(avg) :  0.9999396017356847
'''