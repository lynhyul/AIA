import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split as ts
from xgboost import XGBClassifier

datasets = load_iris()

x = datasets.data
y = datasets.target

print(x.shape)  # 150,4

x_train, x_test, y_train, y_test = ts(x,y,train_size=0.8, shuffle = True, random_state = 110)

pca = PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)   
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]

d = np.argmax(cumsum >= 0.98)+1
print("cumsum >= 0.95", cumsum>=0.98)
print("d :", d)

pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)  # merge fit,transform
x_test = pca.transform(x_test)

model = XGBClassifier(n_jobs=2,use_label_encoder=False)

model.fit(x_train,y_train,eval_metric='logloss')

acc = model.score(x_test,y_test)

print("acc : ",acc)

'''
acc :  0.9666666666666667
'''
