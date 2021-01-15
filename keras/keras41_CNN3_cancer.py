#CNN으로 구성
# 2차원을 4차원으로 늘려서 사용하시오.#CNN으로 구성
# 2차원을 4차원으로 늘려서 사용하시오.


#CNN으로 구성
# 2차원을 4차원으로 늘려서 사용하시오.


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


x_train = x_train.reshape(x_train.shape[0],30,1,1)
x_test = x_test.reshape(x_test.shape[0],30,1,1)


#전처리는 알아서 할 것

#2. modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


model = Sequential()

model.add(Conv2D(filters = 20, kernel_size =(2,2), padding = 'same', strides = 1, 
                                        input_shape=(30,1,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(Conv2D(8,(2,2)))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(315, activation= 'relu'))
model.add(Dense(315, activation= 'relu'))
model.add(Dense(150))
model.add(Dense(50))
model.add(Dense(1, activation= 'sigmoid'))

#compile, fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience= 20, mode = 'auto')
#model.compile(loss = 'mean_squared_error', optimizer='adam', metrics =['accuracy'])
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics =['accuracy','mae'])
model.fit(x_train,y_train, epochs = 300, batch_size=8, validation_split = 0.3, 
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
[loss, accuracy, mae] :  [0.13440221548080444, 0.9649122953414917, 0.09691021591424942]
[[0.9949625  0.9320017  0.9982229  0.00210106 0.97550404 0.52930576
  0.9966302  0.99179924 0.9957744  0.81197083 0.96719855 0.99923265
  0.9957211  0.06946412 0.9623059 ]]
predict =  [1 1 1 0 1 1 1 1 1 1 1 1 1 0 1]
target =  [1 1 1 0 1 0 1 1 1 1 1 1 1 0 1]
'''
