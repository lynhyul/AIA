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

###### 요기 하단때문에 file 분리함


import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer,layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns= ['Layer Type', 'Layer Name', 'Layer Trainable'])


print(aaa)

'''
                                                                            Layer Type Layer Name  Layer Trainable
0  <tensorflow.python.keras.engine.functional.Functional object at 0x000001DC875C4DF0>  vgg16      False
1  <tensorflow.python.keras.layers.core.Flatten object at 0x000001DC87603400>           flatten    True
2  <tensorflow.python.keras.layers.core.Dense object at 0x000001DC875EDC70>             dense      True
3  <tensorflow.python.keras.layers.core.Dense object at 0x000001DC8761B430>             dense_1    True
4  <tensorflow.python.keras.layers.core.Dense object at 0x000001DC87626D30>             dense_2    True
'''