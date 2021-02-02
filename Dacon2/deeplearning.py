import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam



train = pd.read_csv('../data/csv/practice/train.csv')
test = pd.read_csv('../data/csv/practice/test.csv')



temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)

x = temp.iloc[:,3:]/255
x_test = test_df.iloc[:,2:]/255
y = train['digit']
y1 = np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y1[i, digit] = 1

x = x.to_numpy()
x_pred = x_test.to_numpy()

print(x.shape)
print(y.shape)


x = x.reshape(-1,28,28,1)
# pca = PCA(n_components=420)
# x = pca.fit_transform(x)
# x_pred = pca.transform(x_pred)

# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x,y1, test_size=0.2,
#                                             random_state =110) 

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten

# model = Sequential()

# model.add(Conv2D(filters = 512, kernel_size =(4,4), activation='relu', padding = 'same', strides = 1, 
#                                         input_shape=(28,28,1)))
# model.add(Conv2D(filters = 256, kernel_size =(2,2), padding = 'same', activation='relu'))
# model.add(Conv2D(filters = 128, kernel_size =(2,2), padding = 'same', activation='relu'))
# model.add(Conv2D(filters = 64, kernel_size =(2,2), padding = 'same', activation='relu'))
# model.add(Flatten())
# model.add(Dense(256, activation= 'relu'))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dense(128, activation= 'relu'))
# model.add(Dense(32, activation= 'relu'))
# model.add(Dense(10, activation='softmax'))

# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# early_stopping = EarlyStopping(monitor='loss', patience= 30, mode = 'auto')
# lr = ReduceLROnPlateau(monitor='val_acc', patience= 15, mode = 'auto')

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train,y_train, epochs=1000, batch_size=32, validation_split=0.2,  
#                                      callbacks = (early_stopping,lr))

# #4. evaluate , predict

# loss = model.evaluate(x_test,y_test)
# print("loss : ",loss)




