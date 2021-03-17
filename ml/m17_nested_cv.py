#다중분류
# MinMax scaler

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC,SVC

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

#1. data
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

# x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
# shuffle = True, train_size = 0.8 )

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"C" : [1,10,100,1000], "kernel":["linear"]},   # kernel => activation function
    {"C" : [1,10,100], "kernel":["rbf"] , "gamma": [0.001,0.0001] }, # gamma => lr function
    {"C" : [1,10,100,1000],"kernel":["sigmoid"], "gamma":[0.001,0.0001] }
]

#2. modeling

# model = SVC()
model = GridSearchCV(SVC(), parameters, cv=kfold)

score = cross_val_score(model, x, y, cv=kfold)

print("교차검증결과 : ",score)


