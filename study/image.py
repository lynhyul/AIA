from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image as img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# img_data = image.load_img('../data/image/cogi.jpg', target_size=(100,100))

# plt.imshow(img_data)
# plt.show()
data_generator = ImageDataGenerator(rescale = 1./255)   #0~1로 정규화
# # 다중 이미지 불러오기
train_generator = data_generator.flow_from_directory(
    '../data/image/',
    target_size = (50,50),
    batch_size = 1,
    class_mode = 'categorical')
x_train, y_train = train_generator.next()
# 각각의 폴더안에 10개의 이미지 파일이 있고 배치크기를2로
# 정한다면 5번의 배치를 실행 했을 때 1번의 epoch가 실행된다.
# array = img.img_to_array(img_data)/255


# (x_train, y_train) , (x_test, y_test) = array.load_data()

# print(x_train.shape)

# print(train_generator.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten, MaxPooling2D

# model = Sequential()
# model.add(Conv2D(filters =50,kernel_size=(2,2), strides=1, padding='same',input_shape=(50,50,1)))   # 10,10크기(흑백) => (N,10,10,1)
# # model.add(MaxPooling2D(pool_size=3))
# model.add(Conv2D(9,(2,2)))      # padding의 디폴트 => padding = 'valid'
# # model.add(Conv2D(9,(2,3)))
# # model.add(Conv2D(8,2))
# model.add(Flatten())
# model.add(Dense(1))

# model.summary()
