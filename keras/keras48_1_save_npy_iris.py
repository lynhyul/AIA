from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
print(dataset)

print(dataset.keys())
#dict_keys(['data', 'target', 'frame', 'target_names',
#'DESCR', 'feature_names', 'filename'])

print(dataset.frame)
print(dataset.target_names) # [0, 1, 2], ['setosa' 'versicolor' 'virginica']
print(dataset["DESCR"])
print(dataset["feature_names"])
print(dataset.filename)



# x = dataset.data
# y = dataset.target
x_data = dataset['data']
y_data = dataset['target']
print(type(x_data), type(y_data))
np.save('../data/npy/iris_x.npy', arr=x_data)
np.save('../data/npy/iris_y.npy', arr=y_data)
# print(x)
# print(y)
