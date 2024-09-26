# 모델 체크포인트에 시간을 표시해보자


import numpy as np

from tensorflow.keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000,28,28) (60000,)    => 60000,28,28,1
print(x_test.shape, y_test.shape)       # (10000,28,28) (10000,)    => 10000,28,28,1

print(x_train[0])   #    
print("y_train = ", y_train[0])   # 5

print(x_train[0].shape)     #(28, 28)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.         # 이미지의 전처리 (max값이 255기때문에 255로 나눠서
                                                                                            #0~1 사이로 만듦)
x_test = x_test.reshape(10000,28,28,1)/255.
#(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1) ) # x_test = x_train.reshape(10000,28,28,1)/255.

#OneHotEncoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
print(y_train.shape)            # (60000,10)
print(y_test.shape)            # (10000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters = 30, kernel_size =(4,4), padding = 'same', strides = 1, 
                                        input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(8,(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(315, activation= 'relu'))
model.add(Dense(10, activation='softmax'))


import datetime
# date_now = datetime.datetime.now()
# print(date_now)     # 2021-01-27 10:06

# ../data/modelcheckpoint/k_45_0127_1010_{epoch:02d}-{val_loss:.4f}.hdf5


# print(date_time)    # 0127_1044

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
for i in range(5) :
    date_now = datetime.datetime.now()
    date_time = date_now.strftime("%m%d_%H%M%S")
    print(date_time)
    filepath = '../data/modelcheckpoint/'
    filename = '_{epoch:02d}-{val_loss:.4f}.hdf5'
    modelpath = "".join([filepath,"k_45_",date_time,filename])
    early_stopping = EarlyStopping(monitor='val_loss', patience= 5)
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')
#3. compile, fit
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
    hist = model.fit(x_train,y_train, epochs=20, batch_size=128, validation_split=0.2,  
                                     callbacks = [early_stopping, cp])

#4. evaluate , predict

result = model.evaluate(x_test,y_test, batch_size=32)
print("loss : ", result[0])
print("accuracy : ", result[1])



'''
for구문을 쓰지 않아도 에포마다 갱신 할 수 있는 방법이라고 한다. 일단 저장!!
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath='../data/modelcheckpoint/'
filename='_{epoch:02d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "k45_", '{timer}', filename])

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.distribute import distributed_file_utils
@keras_export('keras.callbacks.ModelCheckpoint')
class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
        # `filepath` may contain placeholders such as `{epoch:02d}` and
        # `{mape:.2f}`. A mismatch between logged metrics and the path's
        # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, timer=datetime.datetime.now().strftime('%m%d_%H%M'), **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                        'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath

cp = MyModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
'''