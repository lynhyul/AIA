#다중분류
# MinMax scaler

import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf

from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)      # (150,4)
print(y.shape)      # (150,)

print(x[:5])
print(y)


print(y)
print(x.shape)  # (150,4)
print(y.shape)  # (150,3)

kfold = KFold(n_splits=5, shuffle=True)


#2. modeling

models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
for i in models:
    model = i

    #3. compile fit
    model.fit(x,y)

    #4. evaluation, prediction
    print(f'\n{i}')
    y_pred = model.predict(x)
    # accuracy = accuracy_score(y,y_pred)
    # print('accuracy_score : ', accuracy)
    scores = cross_val_score(model, x,y, cv = kfold)
    print('scores : ', scores)
# scores :  [0.9        0.93333333 0.93333333 0.93333333 1.        ]

'''
Deep learning
loss :  [0.01415738184005022, 1.0]


Machine Learning

SVC()
model_score :  0.9733333333333334
scores :  [0.96666667 1.         1.         0.9        0.96666667]

KNeighborsClassifier()
model_score :  0.9666666666666667
scores :  [1.         0.96666667 1.         0.96666667 0.93333333]

DecisionTreeClassifier()
model_score :  1.0
scores :  [0.93333333 0.93333333 1.         0.93333333 0.9       ]

RandomForestClassifier()
model_score :  1.0
scores :  [1.         0.96666667 0.93333333 1.         0.86666667]

 '''