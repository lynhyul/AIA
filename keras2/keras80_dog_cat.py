# image ../data/image/vgg/
# 강아지, 고양이, 라이언, 슈트 욜케 넣을것
# 파일명 : dog1.jpg, cat1.jpg, lion1.jpg, suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog = load_img('../data/image/vgg/dog1.jfif', target_size=(224,224))
img_cat = load_img('../data/image/vgg/cat1.jpg', target_size=(224,224))
img_lion = load_img('../data/image/vgg/lion1.jfif', target_size=(224,224))
img_suit = load_img('../data/image/vgg/suit1.jfif', target_size=(224,224))

# plt.imshow(img_dog)
# plt.show()

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)

arr_input = np.stack([arr_dog,arr_cat,arr_lion,arr_suit])

print(arr_dog[1][1])
print(type(arr_dog))
from tensorflow.keras.applications.vgg16 import preprocess_input

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)

print(arr_dog[1][1])
print(type(arr_dog))


# Model
model = VGG16()
results = model.predict(arr_input)

print(results)
print('result.shape : ', results.shape)


#Check results
from tensorflow.keras.applications.vgg16 import decode_predictions

decode_results = decode_predictions(results)
print("results[0] : ",decode_results[0])
print("========================================")
print("results[1] : ",decode_results[1])
print("========================================")
print("results[2] : ",decode_results[2])
print("========================================")
print("results[3] : ",decode_results[3])
print("========================================")
'''
results[0] :  [('n02110185', 'Siberian_husky', 0.5978164), ('n02109961', 'Eskimo_dog', 0.3819922), ('n02110063', 'malamute', 0.0072567468), ('n02113186', 'Cardigan', 0.003448518), ('n02114548', 'white_wolf', 0.0025010263)]
========================================
results[1] :  [('n02124075', 'Egyptian_cat', 0.45797604), ('n02123045', 'tabby', 0.4130136), ('n02123159', 'tiger_cat', 0.1273834), ('n02127052', 'lynx', 0.00030834926), ('n04040759', 'radiator', 0.00024303942)]
========================================
results[2] :  [('n03532672', 'hook', 0.35332686), ('n02951585', 'can_opener', 0.23019917), ('n04579432', 'whistle', 0.19786789), ('n04317175', 'stethoscope', 0.05041925), ('n04127249', 'safety_pin', 0.042389903)]
========================================
results[3] :  [('n04350905', 'suit', 0.47582898), ('n02883205', 'bow_tie', 0.28350362), ('n04591157', 'Windsor_tie', 0.19237481), ('n02963159', 'cardigan', 0.026376557), ('n04599235', 'wool', 0.011495415)]
========================================
'''