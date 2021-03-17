#다중분류

import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf

from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target




x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)
# x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)


# scaler = MinMaxScaler()
# scaler.fit(x)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)


#2. modeling

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(10, input_shape=(4,)))
# model.add(Dense(5))
# model.add(Dense(3, activation= 'softmax'))  #다중분류에서는 가지고싶은 결과 수 만큼 입력한다.

models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
for i in models:
    model = i

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print('accuracy_score : ', accuracy)

#결과치 나오게 코딩할것 argmax

'''
Deep learning (Tensorflow)
loss :  [0.01415738184005022, 1.0]

Machine Learning
LinearSVC()
model_score :  0.9666666666666667
accuracy_score :  0.9666666666666667

SVC()
model_score :  0.9666666666666667
accuracy_score :  0.9666666666666667

KNeighborsClassifier()
model_score :  0.9666666666666667
accuracy_score :  0.9666666666666667

DecisionTreeClassifier()
model_score :  0.9666666666666667
accuracy_score :  0.9666666666666667

RandomForestClassifier()
model_score :  0.9666666666666667
accuracy_score :  0.9666666666666667



 '''