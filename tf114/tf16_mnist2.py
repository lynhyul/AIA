# [실습]
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
# tf.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,)

x_train = x_train.reshape(x_train.shape[0], 784)/255.
x_test = x_test.reshape(x_test.shape[0], 784)/255.

y_train = to_categorical(y_train)

x = tf.placeholder(tf.float32, shape = (None, 784))
y = tf.placeholder(tf.float32, shape = (None, 10))

#2. 모델
w1 = tf.Variable(tf.random.normal([784, 256], stddev= 0.1, name = 'weight1')) # stddev는 랜덤수의 범위를 정해주는 것. default는 1.0이다 
b1 = tf.Variable(tf.random.normal([1, 256], stddev= 0.1, name = 'bias1'))     # 0.1로 해주면 적은 범위의 수들이 랜덤하게 뽑힌다.
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)                                    # 초반의 weight를 잡기위해서 stddev를 집어넣은것.

w2 = tf.Variable(tf.random.normal([256, 128], stddev= 0.1, name = 'weight2'))
b2 = tf.Variable(tf.random.normal([1, 128], stddev= 0.1, name = 'bias2'))
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random.normal([128, 64], stddev= 0.1, name = 'weight3'))
b3 = tf.Variable(tf.random.normal([1, 64], stddev= 0.1, name = 'bias3'))
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random.normal([64, 10], stddev= 0.1, name = 'weight4'))
b4 = tf.Variable(tf.random.normal([1, 10], stddev= 0.1, name = 'bias4'))
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

#3. 컴파일, 훈련, 평가
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
# train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)
train = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(131):
        _, cur_loss = sess.run([train, loss], feed_dict = {x:x_train, y:y_train})
        if epoch%10 == 0:
            y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
            y_pred = np.argmax(y_pred, axis = 1)
            print(f'Epoch {epoch}\t===========>\t loss : {cur_loss} \tacc : {accuracy_score(y_test, y_pred)}')

    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis = 1)

    print('accuracy score : ', accuracy_score(y_test, y_pred))\
    # accuracy score :  0.098