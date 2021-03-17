import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split as ts
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv('../data/csv/practice/train.csv')
test = pd.read_csv('../data/csv/practice/test.csv')


from sklearn.model_selection import train_test_split


x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)

x_train = x_train
x_train = x_train.astype('float32')

y_train = train['digit']


x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train,y_train, test_size=0.2, shuffle = True,
                                            random_state = 110) 

x_train1 = x_train1.reshape(-1,28*28)/255
x_test1 = x_test1.reshape(-1,28*28)/255

x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28*28)
x_test = x_test/255

# pca = PCA()
# pca.fit(x_train)
# cumsum = np.cumsum(pca.explained_variance_ratio_)


# print(cumsum)   
'''
d : 277
'''

# d = np.argmax(cumsum >= 0.99999)+1
# print("cumsum >= 0.95", cumsum>=0.99999)
# print("d :", d)

pca = PCA(n_components=752)
x_train1 = pca.fit_transform(x_train1)
x_test1 = pca.transform(x_test1)  # merge fit,transform



# print(y_train.shape)
# print(x_train.shape)
# print(x_test.shape)

model = XGBClassifier(n_estimators=100, learning_rate=0.017,n_jobs=8)
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train1, y_train1, verbose=1, eval_metric='mlogloss')

acc =model.score(x_test1,y_test1)
print(acc)

# predict = model.predict(x_test)
# print(predict)

# # #(1) Save
# # import pickle
# # # pickle.dump(model, open('../data/xgb_save/m39.pikle.data','wb'))
# # # print('save complete')
# # import joblib
# # # joblib.dump(model,'../data/xgb_save/m40.joblib.data')
# # model.save_model("../data/xgb_save/dacon.xgb.model")




# # #2. modeling


# submission = pd.read_csv('../data/csv/practice/submission.csv')
# submission['digit'] = model.predict(x_test)
# submission.head()

# submission.to_csv('../data/csv/practice/baseline.csv', index=False)



# idx = 300
# img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
# digit = train.loc[idx, 'digit']
# letter = train.loc[idx, 'letter']

# plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
# plt.imshow(img)
# plt.show()



# def create_cnn_model(x_train):
#     inputs = tf.keras.layers.Input(x_train.shape[1:])

#     bn = tf.keras.layers.BatchNormalization()(inputs)
#     conv = tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu')(bn)
#     bn = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
#     pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

#     bn = tf.keras.layers.BatchNormalization()(pool)
#     conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
#     bn = tf.keras.layers.BatchNormalization()(conv)
#     conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
#     pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

#     flatten = tf.keras.layers.Flatten()(pool)

#     bn = tf.keras.layers.BatchNormalization()(flatten)
#     dense = tf.keras.layers.Dense(1000, activation='relu')(bn)

#     bn = tf.keras.layers.BatchNormalization()(dense)
#     outputs = tf.keras.layers.Dense(10, activation='softmax')(bn)

#     model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

#     return model
