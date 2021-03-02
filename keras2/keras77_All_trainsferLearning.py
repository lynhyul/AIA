from keras.applications import VGG16, VGG19, Xception
from keras.applications import ResNet101, ResNet101V2, ResNet152V2, ResNet50,ResNet50V2 
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0

model = VGG19()

model.trainable = False

model.summary()






