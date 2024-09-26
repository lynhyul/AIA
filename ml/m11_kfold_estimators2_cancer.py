import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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
 [0.91208791 0.97802198 0.96703297 0.95604396 0.92307692]
BaggingClassifier 의 정답률 : 
 [0.92307692 0.95604396 0.97802198 0.92307692 0.92307692]
BernoulliNB 의 정답률 :
 [0.65934066 0.59340659 0.59340659 0.63736264 0.63736264]
CalibratedClassifierCV 의 정답률 : 
 [0.95604396 0.89010989 0.94505495 0.93406593 0.91208791]
CategoricalNB 은 없는놈!
CheckingClassifier 의 정답률 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는놈!
ComplementNB 의 정답률 :
 [0.86813187 0.93406593 0.9010989  0.86813187 0.92307692]
DecisionTreeClassifier 의 정답률 : 
 [0.93406593 0.96703297 0.87912088 0.95604396 0.9010989 ]
DummyClassifier 의 정답률 :
 [0.46153846 0.62637363 0.50549451 0.50549451 0.53846154]
ExtraTreeClassifier 의 정답률 :
 [0.84615385 0.91208791 0.95604396 0.92307692 0.93406593]
ExtraTreesClassifier 의 정답률 : 
 [0.97802198 0.95604396 0.95604396 0.98901099 0.94505495]
GaussianNB 의 정답률 :
 [0.93406593 0.93406593 0.94505495 0.93406593 0.91208791]
GaussianProcessClassifier 의 정답률 : 
 [0.92307692 0.89010989 0.95604396 0.93406593 0.94505495]
GradientBoostingClassifier 의 정답률 : 
 [0.92307692 0.94505495 0.92307692 0.93406593 0.95604396]
HistGradientBoostingClassifier 의 정답률 : 
 [0.96703297 0.94505495 0.95604396 0.98901099 0.97802198]
KNeighborsClassifier 의 정답률 : 
 [0.96703297 0.91208791 0.92307692 0.9010989  0.94505495]
LabelPropagation 의 정답률 : 
 [0.45054945 0.38461538 0.35164835 0.42857143 0.37362637]
LabelSpreading 의 정답률 : 
 [0.36263736 0.37362637 0.48351648 0.36263736 0.40659341]
LinearDiscriminantAnalysis 의 정답률 :
 [0.94505495 0.96703297 0.91208791 0.94505495 0.94505495]
LinearSVC 의 정답률 : 
 [0.94505495 0.92307692 0.9010989  0.86813187 0.93406593]
LogisticRegression 의 정답률 : 
 [0.91208791 0.92307692 0.9010989  0.96703297 0.93406593]
LogisticRegressionCV 의 정답률 : 
 [0.91208791 0.94505495 0.95604396 0.95604396 0.96703297]
MLPClassifier 의 정답률 : 
 [0.9010989  0.94505495 0.95604396 0.95604396 0.87912088]
MultiOutputClassifier 은 없는놈!
MultinomialNB 의 정답률 :
 [0.89010989 0.91208791 0.87912088 0.87912088 0.92307692]
NearestCentroid 의 정답률 :
 [0.93406593 0.91208791 0.86813187 0.87912088 0.86813187]
NuSVC 의 정답률 : 
 [0.85714286 0.85714286 0.92307692 0.82417582 0.91208791]
OneVsOneClassifier 은 없는놈!
OneVsRestClassifier 은 없는놈!
OutputCodeClassifier 은 없는놈!
PassiveAggressiveClassifier 의 정답률 : 
 [0.82417582 0.89010989 0.92307692 0.91208791 0.74725275]
Perceptron 의 정답률 :
 [0.93406593 0.81318681 0.78021978 0.86813187 0.8021978 ]
QuadraticDiscriminantAnalysis 의 정답률 : 
 [0.91208791 0.91208791 0.95604396 0.96703297 0.96703297]
RadiusNeighborsClassifier 은 없는놈!
RandomForestClassifier 의 정답률 : 
 [0.95604396 0.95604396 0.98901099 0.93406593 0.9010989 ]
RidgeClassifier 의 정답률 :
 [0.93406593 0.95604396 0.93406593 0.97802198 0.94505495]
RidgeClassifierCV 의 정답률 : 
 [0.96703297 0.93406593 0.97802198 0.94505495 0.93406593]
SGDClassifier 의 정답률 : 
 [0.85714286 0.72527473 0.93406593 0.93406593 0.86813187]
SVC 의 정답률 : 
 [0.89010989 0.94505495 0.92307692 0.86813187 0.95604396]
StackingClassifier 은 없는놈!
VotingClassifier 은 없는놈!
'''