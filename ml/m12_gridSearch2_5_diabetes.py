#다중분류
# Model = RandomForestClassifier

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score,r2_score

from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

#1. data
#x, y = load_iris(return_X_y=True)

dataset = load_diabetes()
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

parameters = [
    {"n_estimators" : [100,200,300,400] },
    {'max_depth' : [0,2,4,6,8,10]},
    {'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs' : [-1]}
]

#2. modeling

# model = SVC()
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold)

#3. fit
model.fit(x_train, y_train)

#4. evoluate, predict
print("best parameter : ", model.best_estimator_)



y_pred = model.predict(x_test) # grid serach
print("best score : ",r2_score(y_test, y_pred))



'''
Deep learning
loss :  [0.01415738184005022, 1.0]

best parameter :  RandomForestClassifier()
best score :  0.9666666666666667

 '''