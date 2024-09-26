from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x,y = load_wine(return_X_y=True)
dataset = load_wine()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size = 0.8, shuffle = True, random_state = 110)


#2. modeling
model = XGBClassifier(n_estimators=10, learning_rate=0.017,n_jobs=8,use_label_encoder=False)

#3. fit
model.fit(x_train,y_train, verbose=1, eval_metric= ['mlogloss','merror','cox-nloglik'],   #다중분류 : mlogloss
            eval_set=[(x_train,y_train),(x_test,y_test)],)
aaa = model.score(x_test,y_test)
print("aaa : ", aaa)


result = model.evals_result()
print(result)


import matplotlib.pyplot as plt

epochs = len(result['validation_0']['mlogloss'])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, result['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('M Log loss')
plt.title('XGBoost M Log Loss')
plt.show()

'''
aaa :  0.9166666666666666
{'validation_0': OrderedDict([('mlogloss', [1.075756, 1.053553, 1.03154, 1.010148, 
0.989351, 0.969124, 0.949446, 0.930294, 0.911649, 0.893491])]), 
'validation_1': OrderedDict([('mlogloss', [1.078646, 1.05992, 1.041664, 1.023736, 1.006507, 
0.989357, 0.9729, 0.957056, 0.941857, 0.926479])])}
'''