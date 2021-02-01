import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as ts


datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape)  # 442,10

x_train, x_test, y_train, y_test = ts(x,y,train_size=0.8, shuffle = True, random_state = 110)

pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)  # merge fit,transform
x_test = pca.transform(x_test)

model = RandomForestClassifier()

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc : ",acc)


