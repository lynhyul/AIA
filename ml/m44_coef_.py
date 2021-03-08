x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-5, 63, -21, 9, 1, 45, -3, -9, -49, -27]

print("",x, "\n", y)

import matplotlib.pyplot as plt
plt.plot(x,y)
# plt.show()

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})
print(df)
print(df.shape) # (10, 2)
'''
    X   Y
0  -3  -2
1  31  32
2 -11 -10
3   4   5
4   0   1
5  22  23
6  -2  -1
7  -5  -4
8 -25 -24
9 -14 -13
'''

x_train = df.loc[:,'X']    # 10,
y_train = df.loc[:,'Y']    # 10,

print(x_train.shape, y_train.shape)
print(type(x_train)) # <class 'pandas.core.series.Series'>

x_train = x_train.values.reshape(len(x_train),1) 
print(x_train.shape, y_train.shape) # (10, 1) (10,)
print(type(x_train))    # <class 'numpy.ndarray'>

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print("score: ", score) # score:  1.0

print("기울기(weight) : ",model.coef_)  # [2.]
print("절편(bias) : ",model.intercept_) # 1.0