from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,Activation,ZeroPadding2D,Add
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3

# caltech_dir =  '../data/image/project/'
# categories = ["Beaggle", "Bichon Frise", "Border Collie", "Bulldog","Corgi","Poodle","Retriever","Samoyed",
#                 "Schnauzer","Shih Tzu"]
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
# #1 0 0 0 이면 Beagle
# #0 1 0 0 이면 


# X_train, X_test, y_train, y_test = train_test_split(X, y)
# xy = (X_train, X_test, y_train, y_test)
# np.save("../data/npy/P_project.npy", xy)



# print("ok", len(y))

# print(X_train.shape) # (2442, 255, 255, 3)
# print(X_train.shape[0])   # 2442

idg = ImageDataGenerator(
    width_shift_range=(0.1),   
    height_shift_range=(0.1) 
    )    


X_train, X_test, y_train, y_test = np.load("../data/npy/P_project.npy",allow_pickle=True)
print(X_train.shape)
print(X_train.shape[0])


categories = ["Beaggle", "Bichon Frise", "Border Collie","Bulldog", "Corgi","Poodle","Retriever","Samoyed",
                "Schnauzer","Shih Tzu",]
nb_classes = len(categories)

#일반화
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255 

train_generator = idg.flow(X_train,y_train,batch_size=32)
valid_generator = idg.flow(X_test,y_test)

model = Sequential()
model.add(Conv2D(32, (7,7), padding="same", input_shape=X_train.shape[1:],strides=(2,2), activation='relu'))
model.add(BatchNormalization())                             
model.add(MaxPooling2D((3,3),2))
for i in range (3) :
    model.add(Conv2D(filters = 32, kernel_size =(3,3), padding = 'same', strides=(1,1), activation='relu'))
    model.add(BatchNormalization())                        
    model.add(Conv2D(filters = 32, kernel_size =(3,3), padding = 'valid',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size =(1,1), padding = 'same',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())
    
for i in range (4) :
    model.add(Conv2D(filters = 64, kernel_size =(1,1), padding = 'same',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())                            
    model.add(Conv2D(filters = 64, kernel_size =(3,3), padding = 'valid',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128, kernel_size =(1,1), padding = 'same',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())
    
for i in range (5) :
    model.add(Conv2D(filters = 128, kernel_size =(1,1), padding = 'same',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())                            
    model.add(Conv2D(filters = 128, kernel_size =(3,3), padding = 'valid',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 256, kernel_size =(1,1), padding = 'same',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())
    
for i in range (3) :
    model.add(Conv2D(filters = 128, kernel_size =(1,1), padding = 'same',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())                            
    model.add(Conv2D(filters = 128, kernel_size =(3,3), padding = 'valid',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 256, kernel_size =(1,1), padding = 'same',strides=(1,1), activation='relu'))
    model.add(BatchNormalization())


# model.add(BatchNormalization())
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation= 'relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(64, activation= 'relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001,epsilon=None), metrics=['acc'])


model_path = '../data/modelcheckpoint/Pproject.hdf5'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=60)
lr = ReduceLROnPlateau(patience=30, factor=0.5,verbose=1)

learning_history = model.fit_generator(train_generator,epochs=100, 
validation_data=valid_generator, callbacks=[early_stopping,lr,checkpoint])
    
print((model.evaluate(valid_generator)))