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
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings('ignore')


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
Deep learning
loss :  [0.01415738184005022, 1.0]

LinearSVC()
scores :  [0.91666667 1.         0.95833333 0.875      0.95833333]

SVC()
scores :  [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333]

KNeighborsClassifier()
scores :  [0.91666667 0.95833333 1.         0.95833333 0.95833333]

DecisionTreeClassifier()
scores :  [0.95833333 0.91666667 0.95833333 0.95833333 0.91666667]

RandomForestClassifier()
scores :  [0.91666667 1.         0.91666667 1.         0.91666667]

LogisticRegression()
scores :  [0.95833333 1.         0.95833333 1.         0.91666667]

 '''