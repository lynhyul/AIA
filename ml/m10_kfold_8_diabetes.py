#다중분류
# MinMax scaler

import numpy as np
from sklearn.datasets import load_diabetes
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

dataset = load_diabetes()
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
scores :  [0.4747413  0.32450329 0.57231436 0.40238212 0.41243168]

RandomForestRegressor()
scores :  [0.52246237 0.34390225 0.36601137 0.22897465 0.41408774]

DecisionTreeRegressor()
scores :  [-0.36704037 -0.15431825 -0.39132953 -0.37421302 -0.46603624]

KNeighborsRegressor()
scores :  [0.25201342 0.2633693  0.33773849 0.17452182 0.29569487]
 '''