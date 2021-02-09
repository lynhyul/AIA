import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


x_train = np.load('../data/image/brain/numpy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/numpy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/numpy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/numpy/keras66_test_y.npy')
print(x_train.shape)


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
model.add(Conv2D(filters = 16, kernel_size=(2,2),  activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(2,2),  activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 64, kernel_size=(2,2),  activation= 'relu'))
model.add(Conv2D(filters = 64, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['acc'])
history = model.fit(x_train,y_train, steps_per_epoch=1, epochs= 300,   
                    validation_data= (x_test,y_test), validation_steps=4)
# steps_per_epoch=32 => 32개에 대한 데이터를 1에포에 대해서 32번만 학7bcb29습?

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss'] 
val_loss = history.history['val_loss']


print('acc : ', acc[-1])
print('val_acc : ', val_acc[:-1])