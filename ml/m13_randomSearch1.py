#다중분류
# MinMax scaler

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

#1. data
#x, y = load_iris(return_X_y=True)

# dataset = load_iris()
dataset = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

dataset = dataset.to_numpy()


print(x.shape)      # (150,4)
print(y.shape)      # (150,)

print(x[:5])
print(y)


print(y)
print(x.shape)  # (150,4)
print(y.shape)  # (150,3)

import datetime

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
shuffle = True, train_size = 0.8 )

kfold = KFold(n_splits=5, shuffle=True)

date_now = datetime.datetime.now()
date_time = date_now.strftime("%m월%d일_%H시%M분%S초")
print("start time: ",date_time)

parameters = [
    {"C" : [1,10,100,1000], "kernel":["linear"]},   # kernel => activation function
    {"C" : [1,10,100], "kernel":["rbf"] , "gamma": [0.001,0.0001] }, # gamma => lr function
    {"C" : [1,10,100,1000],"kernel":["sigmoid"], "gamma":[0.001,0.0001] }
]


#2. modeling

# model = SVC()
# model = GridSearchCV(SVC(), parameters, cv=kfold)
model = RandomizedSearchCV(SVC(), parameters, cv=kfold)
#3. fit
model.fit(x_train, y_train)

#4. evoluate, predict
print("best parameter : ", model.best_estimator_)
# best parameter :  SVC(C=1, kernel='linear')


y_pred = model.predict(x_test) # grid serach
print("best score : ",accuracy_score(y_test, y_pred))
# 1.0

date_now = datetime.datetime.now()
date_time = date_now.strftime("%m월%d일_%H시%M분%S초")
print("End time: ",date_time)


'''
Deep learning
loss :  [0.01415738184005022, 1.0]
best parameter :  SVC(C=100, gamma=0.001)
best score :  1.0
'''