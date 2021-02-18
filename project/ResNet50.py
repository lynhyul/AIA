import os
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import tensorflow_hub as hub
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
##########데이터 로드

# caltech_dir =  '../data/image/project/'
# categories = ["0", "1", "2", "3","4","5","6","7",
#                 "8","9"]
# nb_classes = len(categories)

# image_w = 255
# image_h = 255

# pixels = image_h * image_w * 3

# X = []
# y = []

# for idx, cat in enumerate(categories):
    
#     #one-hot 돌리기.
#     label = [0 for i in range(nb_classes)]
#     label[idx] = 1

#     image_dir = caltech_dir + "/" + cat
#     files = glob.glob(image_dir+"/*.jpg")
#     print(cat, " 파일 길이 : ", len(files))
#     for i, f in enumerate(files):
#         img = Image.open(f)
#         img = img.convert("RGB")
#         img = img.resize((image_w, image_h))
#         data = np.asarray(img)

#         X.append(data)
#         y.append(label)

#         if i % 700 == 0:
#             print(cat, " : ", f)

# X = np.array(X)
# y = np.array(y)


x_train, x_test, y_train, y_test = np.load("../data/npy/P_project.npy",allow_pickle=True)


categories = ["Beaggle", "Bichon Frise", "Border Collie","Bulldog", "Corgi","Poodle","Retriever","Samoyed",
                "Schnauzer","Shih Tzu",]
nb_classes = len(categories)


#일반화
x_train = x_train.astype(float) / 255
x_test = x_test.astype(float) / 255


resent = ResNet50(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])

# new_model = Model(inputs = resent, outputs = resent.layers[:-1])
x = resent.output
x = MaxPooling2D(pool_size=(2,2)) (x)
x = Dropout(0.5) (x)
x = Flatten() (x)

x = Dense(128, activation= 'relu') (x)
x = BatchNormalization() (x)
x = Dense(64, activation= 'relu') (x)
x = BatchNormalization() (x)
x = Dropout(0.2) (x)
x = Dense(10, activation= 'softmax') (x)

model = Model(inputs = resent.input, outputs = x)


# for layer_ in model.layers :
#     print(layer_)
#     print(layer_.get_output_at(0).get_shape().as_list())


#false로 적용 했을 때 결과값이 엉망.... true는 적용안했을때와 결과와 속도가 같았다.
# summary결과 또한 적용 안한것과 파라미터가 같은것으로 보아 true가 디폴트인듯 하다

# resent.trainable = True
#위 결과와 똑같았다. 둘의 차이점은 없는듯하다. 다만 for문은 resent.layer[:100]과 같은 방법으로 세세하게 조정이 가능한듯 하다.

# model.summary()

# model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])


# model_path = '../data/modelcheckpoint/Pproject7.hdf5'
# checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=60)
# # lr = ReduceLROnPlateau(patience=30, factor=0.5,verbose=1)

# history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))

# print("정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))

# # 정확도 : 0.9261

# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['acc'])      
# plt.plot(history.history['val_acc'])
# plt.title('loss & acc')
# plt.ylabel('loss & acc')
# plt.xlabel('epoch')
# plt.legend(['tran loss', 'val loss', 'train acc','val acc'])    #주석
# plt.show()
