import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
tf.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,)

x_train = x_train.reshape(x_train.shape[0], 784)/255.
x_test = x_test.reshape(x_test.shape[0], 784)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x = tf.placeholder(tf.float32, shape = (None, 784))
y = tf.placeholder(tf.float32, shape = (None, 10))

#2. 모델
# w1 = tf.Variable(tf.random.normal([784, 256], name = 'weight1'))
w1 = tf.get_variable('weight1', shape = [784, 256], initializer = tf.contrib.layers.xavier_initializer())
print("w1 : ",w1) # w1 :  <tf.Variable 'weight1:0' shape=(784, 256) dtype=float32_ref>
b1 = tf.Variable(tf.random.normal([1,256], name = 'bias1'))    
print("b1 : ",b1) # b1 :  <tf.Variable 'Variable:0' shape=(256,) dtype=float32_ref>
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1) 
print("layer1 : ", layer1) # layer1 :  Tensor("Relu:0", shape=(?, 256), dtype=float32)                      
# # layer1 = tf.nn.dropout(layer1, keep_prob=0.3)

w2 = tf.get_variable('weight2', shape = [256, 128], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random.normal([1,128], name = 'bias2'))    
layer2 = tf.nn.elu(tf.matmul(layer1, w2) + b2) 

w3 = tf.get_variable('weight3', shape = [128, 64], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random.normal([1,64], name = 'bias3'))   
layer3 = tf.nn.selu(tf.matmul(layer2, w3) + b3) 

w4 = tf.get_variable('weight4', shape = [64, 10], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([1,10], name = 'bias4'))   
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

#3. 컴파일, 훈련, 평가
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
# train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)
train = tf.train.AdamOptimizer(learning_rate=0.0023).minimize(loss)

# 배치사이즈와 epoch를 정해준다.
training_epochs = 9
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000/100 = 600개의 loss가 나오는데, 그 loss들을 더해준뒤에 평균값을 낸다.


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_loss = 0

        # 배치 사이즈를 만들어보자
        for i in range(total_batch) :
            start = i * batch_size
            end = start + batch_size

            batch_x, batch_y = x_train[start:end], y_train[start:end]
            feed_dict = {x:batch_x, y:batch_y}
            _, cur_loss = sess.run([train, loss], feed_dict = feed_dict)
            avg_loss += cur_loss/total_batch

            
            print(f'Epoch {epoch} \t===========>\t loss : {avg_loss:.8f}')

    print('훈련 끝')

    prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print('Acc : ', sess.run(accuracy, feed_dict = {x:x_test, y:y_test}))


    # Acc :  0.9734