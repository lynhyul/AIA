# [실습]

import tensorflow as tf
import numpy as np



tf.set_random_seed(66)

datasets = tf.keras.datasets.mnist

(x_train, y_train) , (x_test, y_test) = datasets.load_data()
# print(x_train.shape) # 60000, 28, 28

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

x_train = x_train.reshape(-1,28*28)/255.
x_test = x_test.reshape(-1,28*28)/255.

x = tf.placeholder(tf.float32, shape = [None,28*28])
y = tf.placeholder(tf.float32, shape = [None,10])


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()


w = tf.Variable(tf.zeros([28*28,10], name='weight'))
b = tf.Variable(tf.zeros([1,10]), name = 'bias') # y의 출력의 갯수에 맞게 조정한다.(행렬의 덧셈 방법 참고)
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

# w1 = tf.Variable(tf.zeros([28*28,128], name='weight1'))
# b1 = tf.Variable(tf.zeros([128]), name = 'bias1') # y의 출력의 갯수에 맞게 조정한다.(행렬의 덧셈 방법 참고)
# layer1 = tf.matmul(x,w1) +b1

# w2 = tf.Variable(tf.zeros([128,64], name='weight2'))
# b2 = tf.Variable(tf.zeros([64]), name = 'bias2') # y의 출력의 갯수에 맞게 조정한다.(행렬의 덧셈 방법 참고)
# layer2 = tf.nn.relu(tf.matmul(layer1,w2) +b2)

# w3 = tf.Variable(tf.zeros([64,32], name='weight3'))
# b3 = tf.Variable(tf.zeros([32]), name = 'bias3') # y의 출력의 갯수에 맞게 조정한다.(행렬의 덧셈 방법 참고)
# layer3 = tf.nn.relu(tf.matmul(layer2,w3) +b3)

# w4 = tf.Variable(tf.zeros([32,16], name='weight4'))
# b4 = tf.Variable(tf.zeros([16]), name = 'bias4') # y의 출력의 갯수에 맞게 조정한다.(행렬의 덧셈 방법 참고)
# layer4 = tf.nn.relu(tf.matmul(layer3,w4) +b4)

# w5 = tf.Variable(tf.zeros([64,10], name='weight5'))
# b5 = tf.Variable(tf.zeros([1,10]), name = 'bias5') # y의 출력의 갯수에 맞게 조정한다.(행렬의 덧셈 방법 참고)
# hypothesis = tf.nn.softmax(tf.matmul(layer4,w5) + b5)

# cost = tf.reduce_mean(tf.square(hypothesis -y))
loss = tf.reduce_mean(-tf.reduce_sum(y* tf.log(hypothesis), axis=1)) # categorical_crossentropy
train = tf.train.AdamOptimizer(learning_rate=0.017).minimize(loss)

from sklearn.metrics import accuracy_score

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # 변수 초기화

    for step in range(201) :
        loss_val, _ = sess.run([loss,train], feed_dict = {x:x_train, y:y_train})

        if step % 10 ==0 :
            y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
            y_pred = np.argmax(y_pred, axis= 1)
            print(f'Epoch {step}\t===========>\t loss : {loss_val} \tacc : {accuracy_score(y_test, y_pred)}')

    # predict
    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    # y_test = np.argmax(y_test, axis =1)
    # print("y_pred : ",y_pred)
    # print("y_test : ", y_test)
    print('accuracy_score : ', accuracy_score(y_test, y_pred))
    # accuracy_score :  0.9269