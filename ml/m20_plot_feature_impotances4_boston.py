# DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=110)
print(x.shape)
print(dataset.feature_names)
#2. 모델
model = DecisionTreeRegressor(max_depth=5)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
acc = model.score(x_test,y_test)

print(model.feature_importances_)   # [0.01710498 0.         0.57164695 0.41124807] => feature의 순번

print("acc : ", acc)                # acc :  0.9666666666666667

'''
[4.26536551e-02 0.00000000e+00 2.59100758e-03 1.99828226e-16
 6.92901500e-03 6.01927165e-01 1.80869378e-03 8.21663546e-02
 0.00000000e+00 6.21425851e-03 2.24930167e-02 1.14464267e-02
 2.21770407e-01]
acc :  0.7501992162802863
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model) :
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_dataset(model)
plt.show()