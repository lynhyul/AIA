
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)

kfold = KFold(n_splits=5, shuffle=True) #shuffle은 행을 섞는것.


allAlgorithms = all_estimators(type_filter='classifier')

for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()

        scores = cross_val_score(model, x_train,y_train, cv = kfold)
        print(name, '의 정답률 : \n', scores)
    except :
        print(name, '은 없는놈!')
        

import sklearn
print(sklearn.__version__)  # 0.23.2


'''
Deep learning (Tensorflow) DNN
loss :  [0.01415738184005022, 1.0]

AdaBoostClassifier 의 정답률 : 
 [0.91666667 0.875      0.91666667 0.95833333 1.        ]
BaggingClassifier 의 정답률 : 
 [1.         0.91666667 1.         0.95833333 0.875     ]
BernoulliNB 의 정답률 :
 [0.33333333 0.16666667 0.45833333 0.29166667 0.29166667]
CalibratedClassifierCV 의 정답률 : 
 [0.83333333 0.91666667 0.83333333 0.91666667 0.79166667]
CategoricalNB 의 정답률 :
 [0.91666667 0.83333333 0.95833333 0.91666667 0.95833333]
CheckingClassifier 의 정답률 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는놈!
ComplementNB 의 정답률 : 
 [0.91666667 0.625      0.70833333 0.66666667 0.70833333]
DecisionTreeClassifier 의 정답률 : 
 [0.95833333 0.91666667 0.875      1.         1.        ]
DummyClassifier 의 정답률 :
 [0.29166667 0.16666667 0.25       0.33333333 0.29166667]
ExtraTreeClassifier 의 정답률 : 
 [1.         0.91666667 1.         0.875      0.91666667]
ExtraTreesClassifier 의 정답률 : 
 [0.95833333 0.95833333 0.91666667 0.91666667 1.        ]
GaussianNB 의 정답률 :
 [1.         0.91666667 0.95833333 0.95833333 1.        ]
GaussianProcessClassifier 의 정답률 : 
 [1.         0.875      0.95833333 1.         0.95833333]
GradientBoostingClassifier 의 정답률 : 
 [0.91666667 0.875      1.         0.95833333 0.91666667]
HistGradientBoostingClassifier 의 정답률 : 
 [0.95833333 0.95833333 0.91666667 0.83333333 0.95833333]
KNeighborsClassifier 의 정답률 :
 [0.95833333 0.95833333 0.91666667 0.91666667 1.        ]
LabelPropagation 의 정답률 :
 [1.         0.875      0.91666667 0.95833333 0.95833333]
LabelSpreading 의 정답률 : 
 [0.95833333 0.95833333 0.95833333 0.95833333 0.91666667]
LinearDiscriminantAnalysis 의 정답률 :
 [1.         0.95833333 0.91666667 1.         1.        ]
LinearSVC 의 정답률 : 
 [1.         0.91666667 1.         0.95833333 0.91666667]
LogisticRegression 의 정답률 : 
 [1.         0.95833333 1.         1.         0.91666667]
 LogisticRegressionCV 의 정답률 : 
 [1.         0.91666667 1.         0.95833333 0.95833333]
MLPClassifier 의 정답률 : 
 [1.         1.         0.95833333 0.79166667 0.91666667]
MultiOutputClassifier 은 없는놈!
MultinomialNB 의 정답률 :
 [0.79166667 0.66666667 0.79166667 0.70833333 0.66666667]
NearestCentroid 의 정답률 :
 [0.91666667 0.95833333 0.91666667 0.875      0.91666667]
NuSVC 의 정답률 : 
 [1.         0.91666667 0.875      1.         1.        ]
OneVsOneClassifier 은 없는놈!
OneVsRestClassifier 은 없는놈!
OutputCodeClassifier 은 없는놈!
PassiveAggressiveClassifier 의 정답률 :
 [0.79166667 0.83333333 0.95833333 0.70833333 0.79166667]
Perceptron 의 정답률 :
 [0.70833333 0.875      0.79166667 1.         0.66666667]
QuadraticDiscriminantAnalysis 의 정답률 :
 [0.83333333 1.         1.         0.95833333 0.95833333]
RadiusNeighborsClassifier 의 정답률 : 
 [1.         0.875      1.         0.95833333 0.95833333]
RandomForestClassifier 의 정답률 : 
 [0.83333333 1.         0.95833333 0.91666667 1.        ]
RidgeClassifier 의 정답률 :
 [0.91666667 0.66666667 0.83333333 0.875      0.75      ]
RidgeClassifierCV 의 정답률 : 
 [0.875      0.83333333 0.75       0.70833333 0.79166667]
SGDClassifier 의 정답률 :
 [0.79166667 0.625      0.95833333 0.75       0.91666667]
SVC 의 정답률 : 
 [0.91666667 0.91666667 1.         1.         0.95833333]
StackingClassifier 은 없는놈!
VotingClassifier 은 없는놈!
'''