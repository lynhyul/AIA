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

print(x_train.shape)    # 455,30



#전처리는 알아서 할 것

#2. modeling
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, activation= 'relu', input_shape = (30,) ))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(60, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

#compile, fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 15, mode = 'auto')
#model.compile(loss = 'mean_squared_error', optimizer='adam', metrics =['accuracy'])
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics =['acc'])
hist = model.fit(x_train,y_train, epochs = 300, batch_size=5, validation_split = 0.3, 
                            callbacks = early_stopping)

print(hist)
hist1 = hist.history['loss']
print(np.array(hist1).shape)    #100,
print(hist.history.keys())



loss = model.evaluate(x_test,y_test, batch_size=1)
print("[loss, accuracy] : ",loss)

y_predict = model.predict(x_test[0:15])
y_pred = list(map(int,np.round(y_predict,0)))
y_predict = np.transpose(y_predict)
#y_predict = np.where(y_predict>=0.5,1,y_predict)
y_pred = np.transpose(y_pred)
print(y_predict)
print("predict = ",y_pred)
print("target = ",y_test[0:15])


import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])       #회귀모델이기때문에 acc측정이 힘들다
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
plt.show()


'''
[loss, accuracy, mae] :  [0.12791219353675842, 0.9649122953414917, 0.09085875749588013]
[[1.0000000e+00 8.8559693e-01 1.0000000e+00 3.2461065e-04 9.8169839e-01
  3.4595165e-01 1.0000000e+00 7.7419680e-01 1.0000000e+00 8.8512170e-01
  4.1114306e-01 1.0000000e+00 9.9987423e-01 5.6803753e-03 9.5801550e-01]]
predict =  [1 1 1 0 1 0 1 1 1 1 0 1 1 0 1]
target =  [1 1 1 0 1 0 1 1 1 1 1 1 1 0 1]
'''

'''
[loss, accuracy] :  [0.17664489150047302, 0.9210526347160339]
'''