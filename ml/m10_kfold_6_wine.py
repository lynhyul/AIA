#다중분류
# MinMax scaler

import numpy as np
from sklearn.datasets import load_wine
import tensorflow as tf

from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings('ignore')


#x, y = load_iris(return_X_y=True)

dataset = load_wine()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
shuffle = True, train_size = 0.8 )


kfold = KFold(n_splits=5, shuffle=True)


#2. modeling


# train,test split(x)
# scores = cross_val_score(model, x_train,y, cv = kfold)
# print('scores : ', scores)
# scores :  [0.9        0.93333333 0.93333333 0.93333333 1.        ]
models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
            LogisticRegression()]
for i in models:
    model = i
    print(f'\n{i}')
# train,test split(o)
    scores = cross_val_score(model, x_train,y_train, cv = kfold)
    print('scores : ', scores)
# scores :  [0.875      0.95833333 0.79166667 0.875      1.        ]


'''
LinearSVC()
scores :  [0.89655172 0.89655172 0.89285714 0.67857143 0.85714286]

SVC()
scores :  [0.86206897 0.62068966 0.67857143 0.57142857 0.64285714]

KNeighborsClassifier()
scores :  [0.75862069 0.65517241 0.71428571 0.60714286 0.78571429]

DecisionTreeClassifier()
scores :  [0.93103448 0.82758621 0.92857143 0.96428571 0.82142857]

RandomForestClassifier()
scores :  [1.         1.         0.96428571 1.         1.        ]

LogisticRegression()
scores :  [0.96551724 0.93103448 1.         1.         0.92857143]

 '''