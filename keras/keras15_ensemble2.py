# 원하는 분기만을 출력하도록 하는 ensemble 코드를 작성하자.

import numpy as np
#1. 데이터
x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201),range(411,511), range(100,200)])

y1 = np.array([range(711,811), range(1,101), range(201,301)])


# 결과 (3,100)

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
# 결과 (100,3)


#x1,x2,y1에대한 train_test_split
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train,x2_test, y1_train, y1_test = train_test_split(x1,x2,y1, test_size = 0.2, shuffle = True,
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
output1 = Dense(300) (middle1)
output1 = Dense(70) (output1)
output1 = Dense(100) (output1)
output1 = Dense(3) (output1)


#모델 선언
model = Model(inputs=[input1,input2],outputs=[output1]) # 2개이상은 리스트로 취해서 넣는다.

model.summary()


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1_train,x2_train], y1_train, epochs= 300, batch_size=3,
                    validation_split= 0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test],y1_test, batch_size=1)
print("loss : ", loss)


#loss :  [233.51632690429688, 97.42386627197266, 136.09246826171875, 97.42386627197266, 136.09246826171875]
#model.metrics_name :  ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mse', 'dense_15_mse']


y_predict = model.predict([x1_test, x2_test])

print("====================")
print("y_predict : \n", y_predict)



 #RMSE, R2

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict) :
        return np.sqrt(mean_squared_error(y_test,y_predict,))

print("RMSE : ", RMSE(y1_test,y_predict))


R2 = r2_score(y1_test, y_predict)
print("R2 : ", R2)



'''
loss :  [0.1109372153878212, 0.1109372153878212]
====================
y1_predict :
 [[7.1897998e+02 8.7677565e+00 2.0881369e+02]
 [8.0424036e+02 9.3434578e+01 2.9363397e+02]
 [7.1515057e+02 4.6111574e+00 2.0476143e+02]
 [7.1610950e+02 5.6490679e+00 2.0577415e+02]
 [7.6315100e+02 5.2540401e+01 2.5268123e+02]
 [7.5214203e+02 4.1543438e+01 2.4168536e+02]
 [7.1123004e+02 5.0573826e-01 2.0071965e+02]
 [7.8417212e+02 7.3489212e+01 2.7364838e+02]
 [7.9922839e+02 8.8450531e+01 2.8864182e+02]
 [7.7916638e+02 6.8510406e+01 2.6866119e+02]
 [7.3612885e+02 2.5547876e+01 2.2569131e+02]
 [7.2908130e+02 1.8623569e+01 2.1873500e+02]
 [7.3712964e+02 2.6547571e+01 2.2669093e+02]
 [7.4013214e+02 2.9546753e+01 2.2968980e+02]
 [7.7716412e+02 6.6518860e+01 2.6666626e+02]
 [7.6114941e+02 5.0540993e+01 2.5068202e+02]
 [7.9119226e+02 8.0467270e+01 2.8064041e+02]
 [7.5614532e+02 4.5542358e+01 2.4568391e+02]
 [7.4913953e+02 3.8544266e+01 2.3868648e+02]
 [7.6915601e+02 5.8538776e+01 2.5867908e+02]]
RMSE :  0.3330712588777692
R2 :  0.99985963228726
'''