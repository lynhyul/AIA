# 실습
# 맹그러봐!!!

import numpy as np
import pandas as pd

wine = pd.read_csv('../../data/csv/winequality-white.csv', sep = ';', header = 0)
print(wine.head())
print(wine.shape)
print(wine.describe())

wine_npy = wine.values

print(type(wine_npy))

x = wine_npy[:, :-1]
y = wine_npy[:, -1]

x = wine.drop('quality', axis = 1)
y = wine['quality']

# 카테고리를 7->3개로(상,중,하) 함축해서 정확도를 높인다.
newlist = []
for i in list(y):
    if i <= 4:
        newlist +=[0]
    elif i <= 7:
        newlist +=[1]
    else:
        newlist +=[2]
y = newlist

# print(x.shape, y.shape) # (4898, 11) (4898,)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
print(x_train.shape, x_test.shape) # (3918, 11) (980, 11)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# model = KNeighborsClassifier()
model = RandomForestClassifier()
# model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('score : ', score)