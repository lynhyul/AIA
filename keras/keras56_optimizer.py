import numpy as np

#1. 데이터
x = np.array(range(1,11))
y = np.array(range(1,11))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam


#optimizer = Adam(lr=0.1)
# loss :  [0.0014831674052402377, 0.0014831674052402377] 결과물 :  [[10.94151]]
# optimizer = Adam(lr=0.01)
# loss :  [2.2772894139010125e-13, 2.2772894139010125e-13] 결과물 :  [[10.999999]]
# optimizer = Adam(lr=0.001)
# loss :  [1.66266997457358e-13, 1.66266997457358e-13] 결과물 :  [[11.]]
# optimizer = Adam(lr=0.0001)
# loss :  [1.438156402855384e-07, 1.438156402855384e-07] 결과물 :  [[10.999562]]
# loss :  [1.2398971281114834e-13, 1.2398971281114834e-13] 결과물 :  [[11.000001]]

# optimizer = Adadelta(lr=0.1)
# loss :  [1.2456379408831708e-05, 1.2456379408831708e-05] 결과물 :  [[11.004892]]
#optimizer = Adadelta(lr=0.01)
# loss :  [1.3457618479151279e-05, 1.3457618479151279e-05] 결과물 :  [[10.992271]]
#optimizer = Adadelta(lr=0.001)
# loss :  [5.468192100524902, 5.468192100524902] 결과물 :  [[6.776033]]



# optimizer = Adamax(lr=0.1)
# loss :  [1.1985829928562453e-07, 1.1985829928562453e-07] 결과물 :  [[11.000022]]
optimizer = Adamax(lr=0.01)
# loss :  [6.608047442568932e-13, 6.608047442568932e-13] 결과물 :  [[11.000001]]
# optimizer = Adamax(lr=0.001)
# loss :  [1.720236468827352e-07, 1.720236468827352e-07] 결과물 :  [[11.000357]]

# optimizer = Adagrad(lr=0.1)
# loss :  [21349.07421875, 21349.07421875] 결과물 :  [[-215.06343]]
# optimizer = Adagrad(lr=0.01)
# loss :  [1.9998273259602684e-08, 1.9998273259602684e-08] 결과물 :  [[11.000304]]
# optimizer = Adagrad(lr=0.001)
# loss :  [2.8073196517652832e-05, 2.8073196517652832e-05] 결과물 :  [[11.001221]]


# optimizer = RMSprop(lr=0.1)
# loss :  [6529491968.0, 6529491968.0] 결과물 :  [[89958.984]]
# optimizer = RMSprop(lr=0.01)
# loss :  [19.220182418823242, 19.220182418823242] 결과물 :  [[2.4475565]]
# optimizer = RMSprop(lr=0.001)
# loss :  [0.0014277009759098291, 0.0014277009759098291] 결과물 :  [[10.919306]]
# optimizer = RMSprop(lr=0.0001)
# loss :  [0.000121327604574617, 0.000121327604574617] 결과물 :  [[11.000052]]
# optimizer = RMSprop(lr=0.00001)
# loss :  [4.037881808471866e-05, 4.037881808471866e-05] 결과물 :  [[10.986315]]
# optimizer = RMSprop(lr=0.000001)
# loss :  [20.038387298583984, 20.038387298583984] 결과물 :  [[3.0121744]]

# optimizer = SGD(lr=0.1)
# loss :  [nan, nan] 결과물 :  [[nan]]
# optimizer = SGD(lr=0.01)
# loss :  [nan, nan] 결과물 :  [[nan]]
# optimizer = SGD(lr=0.001)
# loss :  [3.844012042009126e-07, 3.844012042009126e-07] 결과물 :  [[11.000924]]
# optimizer = SGD(lr=0.0001)
# loss :  [0.0014368873089551926, 0.0014368873089551926] 결과물 :  [[10.950289]]

# optimizer = Nadam(lr=0.1)
# loss :  [180472709120.0, 180472709120.0] 결과물 :  [[-608899.94]]
# optimizer = Nadam(lr=0.01)
# loss :  [1.2079226507921703e-13, 1.2079226507921703e-13] 결과물 :  [[10.999999]]
# optimizer = Nadam(lr=0.001)
# oss :  [0.0002715009613893926, 0.0002715009613893926] 결과물 :  [[10.97822]]










model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])   # 정사하강법
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가,예측

loss = model.evaluate(x,y,batch_size=1)
y_pred = model.predict([11])
print("loss : ", loss, "결과물 : ", y_pred)
