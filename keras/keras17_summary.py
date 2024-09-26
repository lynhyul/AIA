#summary()에 대한 이해 (제일중요~)
#실습2 + 과제
#ensenmble1, 2, 3, 4 에 대해 서머리를 계산하고 이해한 것을 과제로 제출 할 것
#layer를 만들 때 'name'에 대해 확인하고 설명 할 것
#얘를 반드시 써야 할 때가 있다. 그때를 말해라.

import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델구성
from tensorflow.keras.models import Sequential, Model   
from tensorflow.keras.layers import Dense, Input   

model = Sequential()
model.add(Dense(50, input_dim=1, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(40))
model.add(Dense(1))

model.summary()


'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
'''

#모델 1
input1 = Input(shape=(3,))
dense1 = Dense(100, activation= 'relu', name = 'input1_1') (input1)
dense1 = Dense(50, activation= 'relu', name = 'input1_2') (dense1)
#output1 = Dense(3) (dense1) 

#모델 2
input2 = Input(shape=(3,))
dense2 = Dense(100, activation= 'relu', name = 'input2_1') (input2)
dense2 = Dense(50, activation= 'relu', name = 'input2_2') (dense2)
dense2 = Dense(50, activation= 'relu', name = 'input2_3') (dense2)
dense2 = Dense(50, activation= 'relu', name = 'input2_4') (dense2)
#output2 = Dense(3) (dense2) 

#모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
#from tensorflow.layers.merge import concatenate, Concatenate
#from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
#concatenate로 합친것또한 layer이다. 아래처럼 layer를 추가해도 되나, 바로 분기해도 된다.
middle1 = Dense(300, name = 'middle_1') (merge1)
middle1 = Dense(100, name = 'middle_2') (middle1)
middle1 = Dense(10, name = 'middle_3') (middle1)

#모델 분기1
output1 = Dense(300, name = 'Output_1') (middle1)
output1 = Dense(70, name = 'Output_2') (output1)
output1 = Dense(100, name = 'Output_3') (output1)
output1 = Dense(3, name = 'Output') (output1)

model2 = Model(inputs = [input1,input2], outputs = [output1])
model2.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 50)                100
_________________________________________________________________
dense_1 (Dense)              (None, 30)                1530
_________________________________________________________________
dense_2 (Dense)              (None, 40)                1240
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 41
=================================================================
Total params: 2,911
Trainable params: 2,911
Non-trainable params: 0
_________________________________________________________________
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
input2_1 (Dense)                (None, 100)          400         input_2[0][0]
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
input2_2 (Dense)                (None, 50)           5050        input2_1[0][0]
__________________________________________________________________________________________________
input1_1 (Dense)                (None, 100)          400         input_1[0][0]
__________________________________________________________________________________________________
input2_3 (Dense)                (None, 50)           2550        input2_2[0][0]
__________________________________________________________________________________________________
input1_2 (Dense)                (None, 50)           5050        input1_1[0][0]
__________________________________________________________________________________________________
input2_4 (Dense)                (None, 50)           2550        input2_3[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 100)          0           input1_2[0][0]
                                                                 input2_4[0][0]
__________________________________________________________________________________________________
middle_1 (Dense)                (None, 300)          30300       concatenate[0][0]
__________________________________________________________________________________________________
middle_2 (Dense)                (None, 100)          30100       middle_1[0][0]
__________________________________________________________________________________________________
middle_3 (Dense)                (None, 10)           1010        middle_2[0][0]
__________________________________________________________________________________________________
Output_1 (Dense)                (None, 300)          3300        middle_3[0][0]
__________________________________________________________________________________________________
Output_2 (Dense)                (None, 70)           21070       Output_1[0][0]
__________________________________________________________________________________________________
Output_3 (Dense)                (None, 100)          7100        Output_2[0][0]
__________________________________________________________________________________________________
Output (Dense)                  (None, 3)            303         Output_3[0][0]
==================================================================================================
Total params: 109,183
Trainable params: 109,183
Non-trainable params: 0
'''