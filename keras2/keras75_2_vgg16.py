from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights='imagenet', include_top=False,input_shape=(32,32,3))

vgg16.trainable = False
'''
Total params: 14,719,879
Trainable params: 5,191
Non-trainable params: 14,714,688
'''

# vgg16.trainable = True
'''
Total params: 14,719,879
Trainable params: 5,191
Non-trainable params: 14,714,688
'''



model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) #, activation= 'softmax'))
model.summary()

print("그냥 가중치의 수 : ", len(model.weights))   #32 -> (weight가 있는 layer * (i(input)bias + o(output)bias))
print("동결 후 훈련되는 가중치의 수 : ",len(model.trainable_weights))   #6
'''
그냥 가중치의 수 :  32
동결 후 훈련되는 가중치의 수 :  6

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 1, 1, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 10)                5130
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 14,719,879
Trainable params: 5,191
Non-trainable params: 14,714,688
_________________________________________________________________
'''