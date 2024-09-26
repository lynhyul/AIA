import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from xgboost import XGBRegressor


datasets = load_boston()

x = datasets.data
y = datasets.target

print(x.shape)  #(506, 13)

# pca = PCA(n_components=7)
# x2 = pca.fit_transform(x)  # merge fit,transform
# print(x2)
# print(x2.shape) # 442,7

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)  
# # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
# print(sum(pca_EVR)) # 0.9479436357350414

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)   
# [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791
#  0.99962696 0.9998755  0.99996089 0.9999917  0.99999835 0.99999992
#  1.        ]

d = np.argmax(cumsum >= 0.95)+1
print("cumsum >= 0.95", cumsum>= 0.95)
print("d :", d)
'''
cumsum >= 0.95 [False  True  True  True  True  True  True  True  True  True  True  True
  True]
d : 2
'''

# model = XGBRegressor(n_jobs=2)
# model.fit(x,y)

# acc = model.score(x,y)
# print(acc)

# #0.999982906820517

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

