import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from xgboost import XGBRegressor


datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape)  # 442,10

# pca = PCA(n_components=7)
# x2 = pca.fit_transform(x)  # merge fit,transform
# print(x2)
# print(x2.shape) # 442,7

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)  
# # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
# print(sum(pca_EVR)) # 0.9479436357350414


# 7 : 0.9479436357350414
# 8 : 0.9913119559917797
# 9 : 0.9991439470098977

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)   
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]

d = np.argmax(cumsum >= 0.95)+1
print("cumsum >= 0.95", cumsum>=0.95)
print("d :", d)
# cumsum >= 0.95 [False False False False False False False  True  True  True]
# d : 8


# model = XGBRegressor(n_jobs=2)
# model.fit(x,y)

# acc = model.score(x,y)
# print(acc)
# # 0.999990274544785

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

