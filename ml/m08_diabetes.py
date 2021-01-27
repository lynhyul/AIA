
import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf

from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score,r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x= dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)

print(np.max(x), np.min(y))
print(dataset.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(dataset.DESCR)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)
# x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)


scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)


#2. modeling

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(10, input_shape=(4,)))
# model.add(Dense(5))
# model.add(Dense(3, activation= 'softmax'))  #다중분류에서는 가지고싶은 결과 수 만큼 입력한다.


# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
model = RandomForestClassifier()
# model = DecisionTreeClassifier()

#3. compile fit

# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience= 5, mode = 'auto')

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train)
# model.fit(x,y)

#4. evaluate , predict

result = model.score(x_test,y_test)
# result = model.score(x,y)
print("result : ",result)


y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2_score : ',r2)

#결과치 나오게 코딩할것 argmax

'''
Machine Learning (train_test_split)

1. LinearSVC
result :  0.0
r2_score :  0.3223264687090769
(2) MinMaxscaler
result :  0.0
r2_score :  0.07451461595772846

2. SVC
result :  0.0
r2_score :  0.21212216289188535
(2) MinMaxscaler
result :  0.0
r2_score :  0.19058546709805102

3. KNeighborsClassifier
result :  0.0
r2_score :  -0.3583804903867607
(2) MinMaxscaler
result :  0.0
r2_score :  -0.6228377982448228

4. RandomForestClassifier
result :  0.0
r2_score :  0.33533949001395646


5. DecisionTreeClassifier
result :  0.011235955056179775
r2_score :  -0.32903687610175325



 '''