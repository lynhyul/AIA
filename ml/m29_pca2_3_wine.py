import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from xgboost import XGBClassifier


datasets = load_wine()

x = datasets.data
y = datasets.target

print(x.shape)  #(178, 13)

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
# [0.99809123 0.99982715 0.99992211 0.99997232 0.99998469 0.99999315
#  0.99999596 0.99999748 0.99999861 0.99999933 0.99999971 0.99999992
#  1.        ]

d = np.argmax(cumsum >= 0.95)+1
print("cumsum >= 0.95", cumsum>= 0.95)
print("d :", d)
'''
cumsum >= 0.95 [ True  True  True  True  True  True  True  True  True  True  True  True
  True]
d : 1
'''

# model = XGBClassifier(n_jobs=2,eval_metric='mlogloss')
# model.fit(x,y)

# acc = model.score(x,y)
# print(acc)
# # 1.0

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

