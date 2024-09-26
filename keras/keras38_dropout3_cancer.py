#유방암 예측 모델을 만들어보자 (이진분류)
#sigmoid에 대해서 조사해보자
#전처리는 알아서 할 것
#실습1. acc를 0.985이상으로 올려볼것
#실습2. predict값 출력해볼것


import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(datasets.DESCR)
print(datasets.feature_names)
print(x.shape)      # 569,30
print(y.shape)      # 569,
print(x[:5])
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 110)




#전처리는 알아서 할 것

#2. modeling
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout

model = Sequential()
model.add(Dense(300, activation= 'relu', input_shape = (30,) ))
model.add(Dense(600, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(800, activation= 'relu'))
model.add(Dense(800, activation= 'relu'))
model.add(Dense(600, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

#compile, fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 15, mode = 'auto')
#model.compile(loss = 'mean_squared_error', optimizer='adam', metrics =['accuracy'])
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics =['accuracy','mae'])
model.fit(x_train,y_train, epochs = 300, batch_size=10, validation_split = 0.3, 
                            callbacks = early_stopping)

loss = model.evaluate(x_test,y_test, batch_size=1)
print("[loss, accuracy, mae] : ",loss)

y_predict = model.predict(x_test[0:15])
y_pred = list(map(int,np.round(y_predict,0)))
y_predict = np.transpose(y_predict)
#y_predict = np.where(y_predict>=0.5,1,y_predict)
y_pred = np.transpose(y_pred)
print(y_predict)
print("predict = ",y_pred)
print("target = ",y_test[0:15])


  


'''
적용 전
[loss, accuracy, mae] :  [0.23702847957611084, 0.8947368264198303, 0.17612338066101074]
[[0.948182   0.5455091  0.99479115 0.0053988  0.7299119  0.2825816
  0.9978307  0.77186185 0.8546791  0.32213497 0.4096725  0.99648106
  0.7750509  0.03607148 0.69941133]]
predict =  [1 1 1 0 1 0 1 1 1 0 0 1 1 0 1]
target =  [1 1 1 0 1 0 1 1 1 1 1 1 1 0 1]

적용 후

[loss, accuracy, mae] :  [0.24583935737609863, 0.9122806787490845, 0.16306540369987488]
[[0.92973393 0.83696204 0.9654475  0.10807542 0.8954599  0.78788525
  0.9448581  0.94994444 0.91675943 0.7224184  0.8488361  0.96094984
  0.93552965 0.38127935 0.86259013]]
predict =  [1 1 1 0 1 1 1 1 1 1 1 1 1 0 1]
target =  [1 1 1 0 1 0 1 1 1 1 1 1 1 0 1]
'''