# 트레인 테스트를 나눈다음에 트레인만 발리데이션 하지말고,
# kfold 한 후에 나온 값을 트레인 테스트 스플릿을 사용하여
# 한 번 더 잘라보자 (즉, 순서를 바꿔서 해봐라)

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

# print(x.shape)      # (150,4)
# print(y.shape)      # (150,)

# print(x[:5])
# print(y)


# print(y)
# print(x.shape)  # (150,4)
# print(y.shape)  # (150,3)

kfold = KFold(n_splits=5, shuffle=True)

print(x.shape,"\n",y.shape) # x = 150,4 / y= 150,

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
shuffle = True, train_size = 0.8 )

print(x_train.shape)    # 120,4

#2. modeling

model = DecisionTreeClassifier()

# train,test split(x)
# scores = cross_val_score(model, x_train,y, cv = kfold)
# print('scores : ', scores)
# scores :  [0.9        0.93333333 0.93333333 0.93333333 1.        ]

# train,test split(o)
scores = cross_val_score(model, x_train,y_train, cv = kfold)
print('scores : ', scores)
# scores :  [0.875      0.95833333 0.79166667 0.875      1.        ]
# scores :  [1.         0.91666667 1.         0.91666667 0.95833333]
