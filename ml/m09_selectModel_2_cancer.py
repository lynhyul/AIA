import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

allAlgorithms = all_estimators(type_filter='classifier')

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
Deep learning (Tensorflow) DNN
[loss, accuracy, mae] :  [0.24958662688732147, 0.9912280440330505, 0.008772376924753189]


AdaBoostClassifier 의 정답률 :  0.9824561403508771
BaggingClassifier 의 정답률 :  0.9649122807017544
BernoulliNB 의 정답률 :  0.631578947368421
CalibratedClassifierCV 의 정답률 :  0.9824561403508771
CategoricalNB 은 없는놈!
CheckingClassifier 의 정답률 :  0.35964912280701755
ClassifierChain 은 없는놈!
ComplementNB 의 정답률 :  0.8157894736842105
DecisionTreeClassifier 의 정답률 :  0.9210526315789473
DummyClassifier 의 정답률 :  0.43859649122807015
ExtraTreeClassifier 의 정답률 :  0.9298245614035088
ExtraTreesClassifier 의 정답률 :  0.9912280701754386
GaussianNB 의 정답률 :  0.9649122807017544
GaussianProcessClassifier 의 정답률 :  0.9824561403508771
GradientBoostingClassifier 의 정답률 :  0.9649122807017544
HistGradientBoostingClassifier 의 정답률 :  0.9824561403508771
KNeighborsClassifier 의 정답률 :  0.9912280701754386
LabelPropagation 의 정답률 :  0.9824561403508771
LabelSpreading 의 정답률 :  0.9824561403508771
LinearDiscriminantAnalysis 의 정답률 :  0.9912280701754386
LinearSVC 의 정답률 :  0.9824561403508771
LogisticRegression 의 정답률 :  0.9824561403508771
LogisticRegressionCV 의 정답률 :  0.9736842105263158
MLPClassifier 의 정답률 :  0.9736842105263158
MultiOutputClassifier 은 없는놈!
MultinomialNB 의 정답률 :  0.8508771929824561
NearestCentroid 의 정답률 :  0.9649122807017544
NuSVC 의 정답률 :  0.9736842105263158
OneVsOneClassifier 은 없는놈!
OneVsRestClassifier 은 없는놈!
OutputCodeClassifier 은 없는놈!
PassiveAggressiveClassifier 의 정답률 :  0.9210526315789473
Perceptron 의 정답률 :  0.956140350877193
QuadraticDiscriminantAnalysis 의 정답률 :  0.9824561403508771
RadiusNeighborsClassifier 은 없는놈!
RandomForestClassifier 의 정답률 :  0.9736842105263158
RidgeClassifier 의 정답률 :  0.9824561403508771
RidgeClassifierCV 의 정답률 :  0.9912280701754386
SGDClassifier 의 정답률 :  0.9736842105263158
SVC 의 정답률 :  0.9824561403508771
StackingClassifier 은 없는놈!
VotingClassifier 은 없는놈!
'''