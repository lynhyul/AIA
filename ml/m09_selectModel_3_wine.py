import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)

allAlgorithms = all_estimators(type_filter='classifier')

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except :
        print(name, '은 없는놈!')
#shift Tap : reverse before push Tab key

import sklearn
print(sklearn.__version__)  # 0.23.2
'''
Deep learning (Tensorflow) CNN
[loss, accuracy, mae] :  [0.010274448432028294, 1.0, 0.00630617793649435]

AdaBoostClassifier 의 정답률 :  0.8611111111111112
BaggingClassifier 의 정답률 :  0.8888888888888888
BernoulliNB 의 정답률 :  0.3611111111111111
CalibratedClassifierCV 의 정답률 :  0.9722222222222222
CategoricalNB 은 없는놈!
CheckingClassifier 의 정답률 :  0.2777777777777778
ClassifierChain 은 없는놈!
ComplementNB 의 정답률 :  0.7777777777777778
DecisionTreeClassifier 의 정답률 :  0.8333333333333334
DummyClassifier 의 정답률 :  0.2222222222222222
ExtraTreeClassifier 의 정답률 :  0.7777777777777778
ExtraTreesClassifier 의 정답률 :  0.9722222222222222
GaussianNB 의 정답률 :  0.9166666666666666
GaussianProcessClassifier 의 정답률 :  0.9444444444444444
GradientBoostingClassifier 의 정답률 :  0.9166666666666666
HistGradientBoostingClassifier 의 정답률 :  0.9166666666666666
KNeighborsClassifier 의 정답률 :  0.9166666666666666
LabelPropagation 의 정답률 :  0.9166666666666666
LabelSpreading 의 정답률 :  0.9166666666666666
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9722222222222222
LogisticRegression 의 정답률 :  0.9444444444444444
LogisticRegressionCV 의 정답률 :  0.9444444444444444
MLPClassifier 의 정답률 :  0.9444444444444444
MultiOutputClassifier 은 없는놈!
MultinomialNB 의 정답률 :  0.8888888888888888
NearestCentroid 의 정답률 :  0.9166666666666666
NuSVC 의 정답률 :  0.9444444444444444
OneVsOneClassifier 은 없는놈!
OneVsRestClassifier 은 없는놈!
OutputCodeClassifier 은 없는놈!
PassiveAggressiveClassifier 의 정답률 :  0.9722222222222222
Perceptron 의 정답률 :  0.9722222222222222
QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
RadiusNeighborsClassifier 의 정답률 :  0.8611111111111112
RandomForestClassifier 의 정답률 :  0.9444444444444444
RidgeClassifier 의 정답률 :  0.9722222222222222
RidgeClassifierCV 의 정답률 :  0.9722222222222222
SGDClassifier 의 정답률 :  0.9444444444444444
SVC 의 정답률 :  0.9722222222222222
StackingClassifier 은 없는놈!
VotingClassifier 은 없는놈!
'''