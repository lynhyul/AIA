# metrics
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

#모델 분기2
output2 = Dense(300) (middle1)
output2 = Dense(70) (output2)
output2 = Dense(70) (output2)
output2 = Dense(3) (output2)

#모델 선언
model = Model(inputs=[input1,input2],outputs=[output1,output2]) # 2개이상은 리스트로 취해서 넣는다.

model.summary()


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae','mse'])
model.fit([x1_train,x2_train], [y1_train,y2_train], epochs= 10, batch_size=3,
                    validation_split= 0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test], batch_size=1)
print("loss : ", loss)


#loss :  [233.51632690429688, 97.42386627197266, 136.09246826171875, 97.42386627197266, 136.09246826171875]
#model.metrics_name :  ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mse', 'dense_15_mse']


y1_predict, y2_predict =model.predict([x1_test, x2_test])

#print("====================")
#print("y1_predict : \n", y1_predict)
#print("====================")
#print("y2_predict : \n",y2_predict)


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
