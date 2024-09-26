import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA


datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape)  # 442,10

pca = PCA(n_components=7)
x2 = pca.fit_transform(x)  # merge fit,transform
print(x2)
print(x2.shape) # 442,7

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)  
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
print(sum(pca_EVR)) # 0.9479436357350414

'''
7 : 0.9479436357350414
8 : 0.9913119559917797
9 : 0.9991439470098977
'''