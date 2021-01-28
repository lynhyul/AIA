#다중분류
# MinMax scaler

import numpy as np
from sklearn.datasets import load_breast_cancer
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

dataset = load_breast_cancer()
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
scores :  [0.93406593 0.91208791 0.91208791 0.91208791 0.83516484]

SVC()
scores :  [0.92307692 0.96703297 0.9010989  0.89010989 0.92307692]

KNeighborsClassifier()
scores :  [0.95604396 0.93406593 0.9010989  0.87912088 0.94505495]

DecisionTreeClassifier()
scores :  [0.91208791 0.91208791 0.9010989  0.93406593 0.94505495]

RandomForestClassifier()
scores :  [0.96703297 0.97802198 0.95604396 0.95604396 0.91208791]

LogisticRegression()
scores :  [0.96703297 0.94505495 0.93406593 0.91208791 0.92307692]

 '''