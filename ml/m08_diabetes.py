
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


models =[DecisionTreeRegressor(), RandomForestRegressor(), KNeighborsRegressor()]
for i in models:
    model = i
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result = model.score(x_test, y_test)
    print(f'\n{i}')
    print("model.score :",result)
    acc = r2_score(y_test, y_pred)
    print("r2_score :",acc)

#결과치 나오게 코딩할것 argmax

'''
Machine Learning (train_test_split)

DecisionTreeRegressor()
model.score : 0.20330666496853378
r2_score : 0.20330666496853378

RandomForestRegressor()
model.score : 0.5680139342130576
r2_score : 0.5680139342130576

KNeighborsRegressor()
model.score : 0.47102341020188654
r2_score : 0.47102341020188654



 '''