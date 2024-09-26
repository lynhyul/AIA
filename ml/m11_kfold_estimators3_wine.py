import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
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
 [0.44827586 1.         0.96428571 0.92857143 0.92857143]
BaggingClassifier 의 정답률 : 
 [0.96551724 1.         0.89285714 0.96428571 0.96428571]
BernoulliNB 의 정답률 :
 [0.27586207 0.37931034 0.39285714 0.25       0.28571429]
CalibratedClassifierCV 의 정답률 : 
 [0.89655172 0.96551724 1.         0.85714286 0.96428571]
CategoricalNB 은 없는놈!
CheckingClassifier 의 정답률 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는놈!
ComplementNB 의 정답률 : 
 [0.75862069 0.55172414 0.71428571 0.82142857 0.60714286]
DecisionTreeClassifier 의 정답률 :
 [0.82758621 0.96551724 0.92857143 0.96428571 0.89285714]
DummyClassifier 의 정답률 : 
 [0.4137931  0.24137931 0.39285714 0.32142857 0.35714286]
ExtraTreeClassifier 의 정답률 :
 [0.89655172 0.93103448 0.78571429 0.82142857 0.85714286]
ExtraTreesClassifier 의 정답률 : 
 [0.96551724 1.         1.         1.         1.        ]
GaussianNB 의 정답률 :
 [0.96551724 1.         0.92857143 1.         1.        ]
GaussianProcessClassifier 의 정답률 : 
 [0.5862069  0.31034483 0.35714286 0.35714286 0.46428571]
GradientBoostingClassifier 의 정답률 : 
 [0.96551724 0.96551724 0.89285714 0.89285714 0.96428571]
HistGradientBoostingClassifier 의 정답률 : 
 [0.96551724 0.96551724 1.         0.89285714 1.        ]
KNeighborsClassifier 의 정답률 :
 [0.62068966 0.82758621 0.64285714 0.78571429 0.75      ]
LabelPropagation 의 정답률 :
 [0.44827586 0.4137931  0.60714286 0.39285714 0.42857143]
LabelSpreading 의 정답률 :
 [0.37931034 0.48275862 0.5        0.42857143 0.35714286]
LinearDiscriminantAnalysis 의 정답률 : 
 [0.96551724 0.93103448 1.         0.92857143 1.        ]
LinearSVC 의 정답률 : 
 [0.89655172 1.         0.85714286 1.         0.75      ]
LogisticRegression 의 정답률 : 
 [0.93103448 0.96551724 1.         0.92857143 1.        ]
LogisticRegressionCV 의 정답률 : 
 [0.96551724 0.93103448 0.92857143 0.96428571 0.92857143]
MLPClassifier 의 정답률 : 
 [0.34482759 0.5862069  0.5        0.25       0.28571429]
MultiOutputClassifier 은 없는놈!
MultinomialNB 의 정답률 :
 [0.82758621 0.89655172 1.         0.78571429 0.89285714]
NearestCentroid 의 정답률 : 
 [0.75862069 0.72413793 0.60714286 0.67857143 0.82142857]
NuSVC 의 정답률 :
 [0.86206897 0.82758621 0.89285714 0.92857143 1.        ]
OneVsOneClassifier 은 없는놈!
OneVsRestClassifier 은 없는놈!
OutputCodeClassifier 은 없는놈!
PassiveAggressiveClassifier 의 정답률 : 
 [0.31034483 0.44827586 0.67857143 0.71428571 0.53571429]
Perceptron 의 정답률 : 
 [0.75862069 0.48275862 0.57142857 0.32142857 0.57142857]
QuadraticDiscriminantAnalysis 의 정답률 :
 [1.         0.96551724 0.92857143 1.         0.96428571]
RadiusNeighborsClassifier 은 없는놈!
RandomForestClassifier 의 정답률 : 
 [0.96551724 1.         1.         0.96428571 0.96428571]
RidgeClassifier 의 정답률 :
 [1.         0.96551724 0.96428571 1.         1.        ]
RidgeClassifierCV 의 정답률 : 
 [0.93103448 1.         1.         1.         1.        ]
SGDClassifier 의 정답률 : 
 [0.34482759 0.72413793 0.64285714 0.53571429 0.75      ]
SVC 의 정답률 :
 [0.62068966 0.5862069  0.67857143 0.71428571 0.85714286]
StackingClassifier 은 없는놈!
VotingClassifier 은 없는놈!
'''