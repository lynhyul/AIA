#concatnate -> Concatnate로 바꿔보자

import numpy as np
#1. 데이터
x1 = np.array([range(100), range(301,401), range(1,101)])
y1 = np.array([range(711,811), range(1,101), range(201,301)])

x2 = np.array([range(101,201),range(411,511), range(100,200)])
y2 = np.array([range(501,601),range(711,811), range(100)])
# 결과 (3,100)

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
# 결과 (100,3)


#x1,y1에대한 train_test_split
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, test_size = 0.2, shuffle = True,
                                            random_state=66)

#x2,y2에대한 train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2, test_size = 0.2, shuffle = True,
                                            random_state=66)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model   
#앞 경로가 중복되는것은 위 처럼 ,로 같이 쓸 수 있다.
from tensorflow.keras.layers import Dense, Input

#모델 1
input1 = Input(shape=(3,))
dense1 = Dense(100, activation= 'relu') (input1)
dense1 = Dense(50, activation= 'relu') (dense1)
#output1 = Dense(3) (dense1) 

#모델 2
input2 = Input(shape=(3,))
dense2 = Dense(100, activation= 'relu') (input2)
dense2 = Dense(50, activation= 'relu') (dense2)
dense2 = Dense(50, activation= 'relu') (dense2)
dense2 = Dense(50, activation= 'relu') (dense2)
#output2 = Dense(3) (dense2) 

#모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
#from tensorflow.layers.merge import concatenate, Concatenate
#from keras.layers import concatenate, Concatenate
merge1 = Concatenate()([dense1, dense2])
#concatenate로 합친것또한 layer이다. 아래처럼 layer를 추가해도 되나, 바로 분기해도 된다.
middle1 = Dense(300) (merge1)
middle1 = Dense(100) (middle1)
middle1 = Dense(10) (middle1)

#모델 분기1
output1 = Dense(300) (middle1)
output1 = Dense(70) (output1)
output1 = Dense(100) (output1)
output1 = Dense(3) (output1)

#모델 분기2
output2 = Dense(300) (middle1)
output2 = Dense(70) (output2)
output2 = Dense(70) (output2)
output2 = Dense(3) (output2)

#모델 선언
model = Model(inputs=[input1,input2],outputs=[output1,output2]) # 2개이상은 리스트로 취해서 넣는다.

model.summary()


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1_train,x2_train], [y1_train,y2_train], epochs= 300, batch_size=3,
                    validation_split= 0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test], batch_size=1)
print("loss : ", loss)


#loss :  [233.51632690429688, 97.42386627197266, 136.09246826171875, 97.42386627197266, 136.09246826171875]
#model.metrics_name :  ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mse', 'dense_15_mse']


y1_predict, y2_predict =model.predict([x1_test, x2_test])

print("====================")
print("y1_predict : \n", y1_predict)
print("====================")
print("y2_predict : \n",y2_predict)

'''
y1_predict :
 [[719.41376     9.211669  209.42863  ]
 [805.2324     94.21451   294.5889   ]
 [715.4092      5.2317615 205.44452  ]
 [716.41034     6.2267413 206.44054  ]
 [763.78687    53.16834   253.46269  ]
 [752.6686     42.158623  242.42969  ]
 [710.42395     1.504972  201.36644  ]
 [785.0116     74.19265   274.5283   ]
 [800.17065    89.22197   289.58047  ]
 [779.9586     69.18285   269.5109   ]
 [736.4978     26.14903   226.38773  ]
 [729.4279     19.162277  219.3897   ]
 [737.5082     27.148088  227.38829  ]
 [740.5399     30.148039  230.3937   ]
 [777.93726    67.18065   267.50467  ]
 [761.7654     51.166523  251.45668  ]
 [792.0859     81.206314  281.55264  ]
 [756.7116     46.162136  246.44167  ]
 [749.6365     39.15598   239.42067  ]
 [769.8512     59.17361   259.48065  ]]
====================
y2_predict :
 [[5.0950626e+02 7.1928937e+02 8.2576447e+00]
 [5.9500067e+02 8.0504877e+02 9.3314026e+01]
 [5.0551376e+02 7.1528058e+02 4.2732754e+00]
 [5.0651187e+02 7.1628278e+02 5.2693434e+00]
 [5.5371979e+02 7.6365277e+02 5.2244244e+01]
 [5.4264227e+02 7.5254211e+02 4.1225624e+01]
 [5.0095132e+02 7.1033679e+02 5.5579734e-01]
 [5.7486578e+02 7.8485876e+02 7.3283432e+01]
 [5.8996594e+02 7.9999500e+02 8.8318748e+01]
 [5.6983246e+02 7.7981348e+02 6.8271660e+01]
 [5.2652991e+02 7.3638000e+02 2.5204798e+01]
 [5.1948895e+02 7.2931226e+02 1.8218737e+01]
 [5.2753674e+02 7.3739056e+02 2.6203941e+01]
 [5.3055768e+02 7.4042139e+02 2.9205227e+01]
 [5.6781854e+02 7.7779370e+02 6.6267998e+01]
 [5.5170575e+02 7.6163275e+02 5.0240849e+01]
 [5.8191254e+02 7.9192236e+02 8.0299911e+01]
 [5.4667035e+02 7.5658234e+02 4.5232349e+01]
 [5.3962109e+02 7.4951190e+02 3.8220516e+01]
 [5.5976208e+02 7.6971313e+02 5.8254429e+01]]
 '''

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
loss :  [0.7783256769180298, 0.31306326389312744, 0.4652623236179352, 0.31306326389312744, 0.4652623236179352]
====================
y1_predict :
 [[7.1801935e+02 8.4554691e+00 2.0814835e+02]
 [8.0492371e+02 9.3831337e+01 2.9386688e+02]
 [7.1382043e+02 4.5923324e+00 2.0418108e+02]
 [7.1487164e+02 5.5565352e+00 2.0517215e+02]
 [7.6297455e+02 5.2650177e+01 2.5256946e+02]
 [7.5180365e+02 4.1507645e+01 2.4143805e+02]
 [7.0961981e+02 7.3658907e-01 2.0021820e+02]
 [7.8430066e+02 7.3922218e+01 2.7382040e+02]
 [7.9964764e+02 8.8939507e+01 2.8889044e+02]
 [7.7922308e+02 6.8857483e+01 2.6876071e+02]
 [7.3546075e+02 2.5342951e+01 2.2523804e+02]
 [7.2827899e+02 1.8389284e+01 2.1820111e+02]
 [7.3648688e+02 2.6336344e+01 2.2624335e+02]
 [7.3957336e+02 2.9323809e+01 2.2926346e+02]
 [7.7719196e+02 6.6831566e+01 2.6673676e+02]
 [7.6094348e+02 5.0624268e+01 2.5054559e+02]
 [7.9140948e+02 8.1012939e+01 2.8090402e+02]
 [7.5586584e+02 4.5559502e+01 2.4548586e+02]
 [7.4875714e+02 3.8468796e+01 2.3840224e+02]
 [7.6906775e+02 5.8727905e+01 2.5864120e+02]]
====================
y2_predict :
 [[ 5.0861591e+02  7.1776361e+02  7.0476532e+00]
 [ 5.9455121e+02  8.0418280e+02  9.2336060e+01]
 [ 5.0448758e+02  7.1356079e+02  3.1331460e+00]
 [ 5.0552002e+02  7.1461310e+02  4.1105022e+00]
 [ 5.5329797e+02  7.6249438e+02  5.1301224e+01]
 [ 5.4226147e+02  7.5138385e+02  4.0191185e+01]
 [ 5.0036105e+02  7.0935431e+02 -7.7501309e-01]
 [ 5.7436761e+02  7.8370532e+02  7.2511292e+01]
 [ 5.8944580e+02  7.9895319e+02  8.7472427e+01]
 [ 5.6935120e+02  7.7865515e+02  6.7461273e+01]
 [ 5.2605676e+02  7.3514331e+02  2.4054317e+01]
 [ 5.1887524e+02  7.2798700e+02  1.7051600e+01]
 [ 5.2708276e+02  7.3616577e+02  2.5054720e+01]
 [ 5.3016235e+02  7.3923340e+02  2.8047920e+01]
 [ 5.6734442e+02  7.7663507e+02  6.5441238e+01]
 [ 5.5129138e+02  7.6047424e+02  4.9281235e+01]
 [ 5.8139087e+02  7.9077570e+02  7.9581291e+01]
 [ 5.4627478e+02  7.5542407e+02  4.4231186e+01]
 [ 5.3925146e+02  7.4835370e+02  3.7161156e+01]
 [ 5.5931787e+02  7.6855475e+02  5.7361221e+01]]
RMSE1 :  0.5595368375542586
RMSE2 :  0.6821089769126246
RMSE(avg) :  0.6208229072334417
R2_1 :  0.9996038585616973
R2_2 :  0.9994112913236793
R2(avg) :  0.9995075749426883
'''