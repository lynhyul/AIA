import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip = True,
    # vertical_flip= True,
    width_shift_range=(-1,1),
    height_shift_range=(-1,1),
    # rotation_range= 5,
    # zoom_range= 1.2,
    # shear_range= 0.7,
    # fill_mode = 'nearest'   #최근접기법, 패딩
)

'''
- rotation_range: 이미지 회전 범위 (degrees)
- width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 
                                (원본 가로, 세로 길이에 대한 비율 값)
- rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 
            모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
- shear_range: 임의 전단 변환 (shearing transformation) 범위
- zoom_range: 임의 확대/축소 범위
- horizontal_flip`: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 
    원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
- fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''
test_datagen = ImageDataGenerator(rescale=1./255)   #test data는 따로 튜닝하지 않고 전처리만 해준다.

# flow or flow_from_directory


xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',        # 사진 160장 (80장 + 80장)
    target_size = (150,150),
    batch_size= 200,  # 160, 150, 150, 1
    class_mode='binary' # 앞에 놈이 0, 뒤에 놈이 1

)

xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size = (150,150),
    batch_size=200,
    class_mode='binary' # 앞에 놈이 0, 뒤에 놈이 1

)

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000002080FFD75E0>

print(xy_train[0])
print(xy_train[0][0].shape) # (160, 150, 150, 3) = x
print(xy_train[0][1].shape) # (160,) = y

# np.save('../data/image/brain/numpy/keras66_train_x.npy', arr=xy_train[0][0])
# np.save('../data/image/brain/numpy/keras66_train_y.npy', arr=xy_train[0][1])
# np.save('../data/image/brain/numpy/keras66_test_x.npy', arr=xy_test[0][0])
# np.save('../data/image/brain/numpy/keras66_test_y.npy', arr=xy_test[0][1])

np.save('../data/image/brain/numpy/keras66_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/brain/numpy/keras66_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/brain/numpy/keras66_test_x.npy', arr=xy_test[0][0])
np.save('../data/image/brain/numpy/keras66_test_y.npy', arr=xy_test[0][1])


x_train = np.load('../data/image/brain/numpy/keras66_train_x.npy')
x_test = np.load('../data/image/brain/numpy/keras66_test_x.npy')

print(x_train.shape)