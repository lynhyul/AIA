#다중분류
# MinMax scaler

import numpy as np
from sklearn.datasets import load_boston
import tensorflow as tf

from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings('ignore')


#x, y = load_iris(return_X_y=True)

dataset = load_boston()
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
models = [LinearRegression(),RandomForestRegressor(), DecisionTreeRegressor(), KNeighborsRegressor()]
for i in models:
    model = i
    print(f'\n{i}')
# train,test split(o)
    scores = cross_val_score(model, x_train,y_train, cv = kfold)
    print('scores : ', scores)
# scores :  [0.875      0.95833333 0.79166667 0.875      1.        ]


'''
LinearRegression()
scores :  [0.68000859 0.67897677 0.80253807 0.670791   0.77305864]

RandomForestRegressor()
scores :  [0.86757714 0.89490308 0.9332648  0.69718659 0.86699091]

DecisionTreeRegressor()
scores :  [0.67137425 0.7948131  0.77854545 0.80864087 0.82854936]

KNeighborsRegressor()
scores :  [0.65686681 0.61207526 0.56282827 0.38470078 0.44451659]

 '''