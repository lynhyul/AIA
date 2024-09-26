from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=110)

print(x.shape)
print(dataset.feature_names)

#2. 모델
model = DecisionTreeClassifier(max_depth=5)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
acc = model.score(x_test,y_test)

print(model.feature_importances_)   # [0.01710498 0.         0.57164695 0.41124807] => feature의 순번

print("acc : ", acc)                # acc :  0.9666666666666667

'''
[0.01681732 0.01681732 0.04675543 0.91960993]
acc :  0.9666666666666667
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