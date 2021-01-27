import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
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
loss :  [0.01415738184005022, 1.0]


AdaBoostClassifier 의 정답률 :  0.9333333333333333
BaggingClassifier 의 정답률 :  0.9333333333333333
BernoulliNB 의 정답률 :  0.16666666666666666
CalibratedClassifierCV 의 정답률 :  0.8333333333333334
CategoricalNB 의 정답률 :  0.13333333333333333
CheckingClassifier 의 정답률 :  0.3
ClassifierChain 은 없는놈!
ComplementNB 의 정답률 :  0.43333333333333335
DecisionTreeClassifier 의 정답률 :  0.9666666666666667
DummyClassifier 의 정답률 :  0.16666666666666666
ExtraTreeClassifier 의 정답률 :  0.9666666666666667
ExtraTreesClassifier 의 정답률 :  0.9333333333333333
GaussianNB 의 정답률 :  0.9333333333333333
GaussianProcessClassifier 의 정답률 :  0.8666666666666667
GradientBoostingClassifier 의 정답률 :  0.9666666666666667
HistGradientBoostingClassifier 의 정답률 :  0.9666666666666667
KNeighborsClassifier 의 정답률 :  0.9666666666666667
LabelPropagation 의 정답률 :  0.9666666666666667
LabelSpreading 의 정답률 :  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9
LogisticRegression 의 정답률 :  0.8666666666666667
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  0.8333333333333334
MultiOutputClassifier 은 없는놈!
MultinomialNB 의 정답률 :  0.43333333333333335
NearestCentroid 의 정답률 :  0.9333333333333333
NuSVC 의 정답률 :  1.0
OneVsOneClassifier 은 없는놈!
OneVsRestClassifier 은 없는놈!
OutputCodeClassifier 은 없는놈!
PassiveAggressiveClassifier 의 정답률 :  0.8333333333333334
Perceptron 의 정답률 :  0.5
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.43333333333333335
RandomForestClassifier 의 정답률 :  0.9666666666666667
RidgeClassifier 의 정답률 :  0.6666666666666666
RidgeClassifierCV 의 정답률 :  0.7666666666666667
SGDClassifier 의 정답률 :  0.9
SVC 의 정답률 :  1.0
StackingClassifier 은 없는놈!
VotingClassifier 은 없는놈!
'''