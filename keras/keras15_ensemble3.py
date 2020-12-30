#실습 2:3 앙상블을 만들어보시오.

import numpy as np
#1. 데이터
x1 = np.array([range(100), range(301,401), range(1,101)])
y1 = np.array([range(711,811), range(1,101), range(201,301)])

x2 = np.array([range(101,201),range(411,511), range(100,200)])
y2 = np.array([range(501,601),range(711,811), range(100)])
y3 = np.array([range(601,701),range(811,911), range(1100,1200)])
# 결과 (3,100)

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)
# 결과 (100,3)


#x1,y1에대한 train_test_split
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, test_size = 0.2, shuffle = True,
                                            random_state=66)

#x2,y2에대한 train_test_split
x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x2,y2,y3, test_size = 0.2, shuffle = True,
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
merge1 = concatenate([dense1, dense2])
#concatenate로 합친것또한 layer이다. 아래처럼 layer를 추가해도 되나, 바로 분기해도 된다.
middle1 = Dense(300) (merge1)
middle1 = Dense(100) (middle1)
middle1 = Dense(10) (middle1)

#모델 분기1
output1 = Dense(300, activation= 'relu') (middle1)
for i in range (1,5) : 
    output1 = Dense(500-i*80, activation= 'relu') (output1)
output1 = Dense(3) (output1)

#모델 분기2
output2 = Dense(300, activation= 'relu') (middle1)
for i in range (1,5) : 
    output2 = Dense(500-i*80, activation= 'relu') (output2)
output2 = Dense(3) (output2)

output3 = Dense(300, activation= 'relu') (middle1)
for i in range (1,5) : 
    output3 = Dense(500-i*80, activation= 'relu') (output3)
output3 = Dense(3) (output3)

#모델 선언
model = Model(inputs=[input1,input2],outputs=[output1,output2,output3]) # 2개이상은 리스트로 취해서 넣는다.

model.summary()


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1_train,x2_train], [y1_train,y2_train,y3_train], epochs= 300, batch_size=5,
                    validation_split= 0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test,y3_test], batch_size=1)
print("loss : ", loss)


#loss :  [233.51632690429688, 97.42386627197266, 136.09246826171875, 97.42386627197266, 136.09246826171875]
#model.metrics_name :  ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mse', 'dense_15_mse']


y1_predict, y2_predict,y3_predict = model.predict([x1_test, x2_test])

print("====================")
print("y1_predict : \n", y1_predict)
print("====================")
print("y2_predict : \n",y2_predict)
print("====================")
print("y3_predict : \n",y3_predict)


 #RMSE, R2

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict) :
        return np.sqrt(mean_squared_error(y_test,y_predict,))
RMSE1 = RMSE(y1_test,y1_predict)
RMSE2 = RMSE(y2_test,y2_predict)
RMSE3 = RMSE(y3_test,y3_predict)

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE(avg) : ", (RMSE3+RMSE2+RMSE1)/3)

R2_1 = r2_score(y1_test, y1_predict)
R2_2 = r2_score(y2_test, y2_predict)
R2_3 = r2_score(y3_test, y3_predict)
print("R2_1 : ", R2_1)
print("R2_2 : ", R2_2)
print("R2_3 : ", R2_3)
print("R2(avg) : ", (R2_3+R2_1+R2_2)/3)


'''
loss :  [1.9705454111099243, 0.2609597146511078, 0.41618290543556213, 1.2934027910232544, 0.2609597146511078, 0.41618290543556213, 1.2934027910232544]
====================
y1_predict :
 [[718.70795     8.785226  208.65187  ]
 [803.7841     94.23959   294.08923  ]
 [714.12146     5.2150965 204.73293  ]
 [715.4754      6.001443  205.69945  ]
 [762.7536     53.073082  253.00853  ]
 [752.17377    41.674202  241.94786  ]
 [708.69025     2.7142956 201.3373   ]
 [783.3974     73.34813   273.39722  ]
 [797.9128     89.1104    288.78998  ]
 [778.4162     68.60477   268.5878   ]
 [736.1712     25.799273  226.09676  ]
 [729.27325    18.808336  219.04146  ]
 [737.1718     26.802462  227.10555  ]
 [740.17676    29.816492  230.15547  ]
 [776.40985    66.735275  266.69775  ]
 [760.8084     50.929737  250.9544   ]
 [790.25055    80.559784  280.50845  ]
 [755.9858     45.766876  245.93478  ]
 [749.3308     38.67131   239.00957  ]
 [768.5878     59.17233   258.98273  ]]
====================
y2_predict :
 [[509.65207   719.7936      7.515367 ]
 [594.5969    804.8269     92.6151   ]
 [505.50443   715.43774     3.871923 ]
 [506.6774    716.75256     4.696283 ]
 [553.7014    763.80237    52.05088  ]
 [542.7093    753.0865     40.305244 ]
 [501.0088    710.1006      1.3861327]
 [574.2856    784.2479     72.456856 ]
 [588.97394   798.8606     87.90181  ]
 [569.39844   779.3799     67.67705  ]
 [526.63245   736.91315    24.36632  ]
 [519.7542    730.0603     17.44244  ]
 [527.6267    737.9061     25.355814 ]
 [530.6377    740.96326    28.284624 ]
 [567.4275    777.4358     65.75222  ]
 [551.7028    761.854      49.91533  ]
 [581.17395   791.03827    79.7341   ]
 [546.7064    756.98315    44.5764   ]
 [539.7428    750.1826     37.164993 ]
 [559.6409    769.631      58.135216 ]]
====================
y3_predict :
 [[ 609.8346   819.3941  1107.4897 ]
 [ 692.3736   903.5157  1194.2886 ]
 [ 605.35187  814.67755 1102.3273 ]
 [ 606.63965  816.1182  1104.0005 ]
 [ 653.0245   862.74225 1152.0542 ]
 [ 642.729    852.5921  1142.0607 ]
 [ 600.2485   808.78424 1095.2671 ]
 [ 672.76154  883.2425  1173.4062 ]
 [ 686.81537  897.48553 1187.6892 ]
 [ 668.0588   878.34424 1168.2972 ]
 [ 627.28143  837.1014  1126.0645 ]
 [ 620.3964   830.1137  1118.7202 ]
 [ 628.2696   838.10535 1127.1215 ]
 [ 631.25116  841.13    1130.3049 ]
 [ 666.175    876.3691  1166.2188 ]
 [ 651.15796  860.8948  1150.2275 ]
 [ 679.3349   889.89667 1180.0756 ]
 [ 646.4891   856.27704 1145.6775 ]
 [ 640.0079   849.93353 1139.4434 ]
 [ 658.63995  868.4686  1157.905  ]]
RMSE1 :  0.5108259393323356
RMSE2 :  0.6451010245886076
RMSE3 :  1.1372637848800853
RMSE(avg) :  0.7643969162670095
R2_1 :  0.9996698291021193
R2_2 :  0.9994734393881964
R2_3 :  0.9983635025778557
R2(avg) :  0.9991689236893905
'''
