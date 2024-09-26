
import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(66)

tf.compat.v1.disable_eager_execution()
# print(tf.executing_eagerly())   # False

# print(tf.__version__)   # 1.14.0 / 2.4.0


# 1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32') /255.
x_test = x_test.reshape(10000,28,28,1).astype('float32') /255.

learning_rate = 0.0001
training_epochs = 30
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000/100

x = tf.compat.v1.placeholder(tf.float32, shape = [None,28,28,1])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,10])

# 2. 모델 구성
# Conv2D summary(filter, kernel_size, input_shape) 파라미터 개수?
# Conv2D(10, (3,3), input_shape=(7,7,1)) => (input_dim x kernal_size + bias) x filter => (1x (3x3) +1) x7 = 70

#L1.
w1 = tf.compat.v1.get_variable('w1', shape = [3, 3, 1, 64])       # 3,3 => 커널사이즈 1 => 흑백, 32 => filter
L1 = tf.nn.conv2d(x,w1, strides=[1,1,1,1], padding='VALID')  # shape 유지
print(L1) # Tensor("Conv2D:0", shape=(None, 26, 26, 64), dtype=float32)
L1 = tf.compat.v1.layers.batch_normalization(L1)
L1 = tf.nn.relu(L1)
print(L1) # Tensor("Relu:0", shape=(None, 26, 26, 64), dtype=float32)

#L2.
w2 = tf.compat.v1.get_variable('w2', shape = [3, 3, 64, 64])  
L2 = tf.nn.conv2d(L1,w2, strides=[1,1,1,1], padding='VALID') 
L2 = tf.compat.v1.layers.batch_normalization(L2)
L2 = tf.nn.relu(L2)
print(L2) # Tensor("Relu_1:0", shape=(None, 24, 24, 64), dtype=float32)

#L3.
w3 = tf.compat.v1.get_variable('w3', shape = [3, 3, 64, 128])  
L3 = tf.nn.conv2d(L2,w3, strides=[1,1,1,1], padding='SAME')  # ?,7,7,32
L3 = tf.compat.v1.layers.batch_normalization(L3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3) # Tensor("MaxPool2d:0", shape=(None, 12, 12, 128), dtype=float32)


#L4.
w4 = tf.compat.v1.get_variable('w4', shape = [3, 3, 128, 128])       
L4 = tf.nn.conv2d(L3,w4, strides=[1,1,1,1], padding='VALID')
L4 = tf.compat.v1.layers.batch_normalization(L4)  
L4 = tf.nn.relu(L4)
print(L4) # 

#L5.
w5 = tf.compat.v1.get_variable('w5', shape = [3, 3, 128, 128])  
L5 = tf.nn.conv2d(L4,w5, strides=[1,1,1,1], padding='VALID') 
L5 = tf.compat.v1.layers.batch_normalization(L5)
L5 = tf.nn.relu(L5)
print(L5) 

#L6.
w6 = tf.compat.v1.get_variable('w6', shape = [3, 3, 128, 256])  
L6 = tf.nn.conv2d(L5,w6, strides=[1,1,1,1], padding='SAME')
L6 = tf.compat.v1.layers.batch_normalization(L6) 
L6 = tf.nn.relu(L6)
L6 = tf.nn.max_pool(L6, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L6) # Tensor("MaxPool2d_1:0", shape=(None, 4, 4, 256), dtype=float32)

#Flatten
L_flat = tf.reshape(L6,[-1,4*4*256]) # 4차원 => 2차원
print("Flatten : ",L_flat)  # Flatten :  Tensor("Reshape:0", shape=(?, 128), dtype=float32)

#L7.
w7 = tf.compat.v1.get_variable('w7', shape = [4*4*256, 128], initializer=tf.initializers.he_normal())
b7 = tf.Variable(tf.compat.v1.random_normal([128]), name='b7')
L7 = tf.matmul(L_flat,w7) +b7
L7 = tf.compat.v1.layers.batch_normalization(L7) 
L7 = tf.nn.relu(L7)
print(L7) # Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

#L8.
w8 = tf.compat.v1.get_variable('w8', shape = [128, 64], initializer=tf.initializers.he_normal())
b8 = tf.Variable(tf.compat.v1.random_normal([64]), name='b8')
L8 = tf.matmul(L7,w8) +b8
L8 = tf.compat.v1.layers.batch_normalization(L8) 
L8 = tf.nn.relu(L8)
print(L6) # Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

#L9.
w9 = tf.compat.v1.get_variable('w9', shape = [64, 32], initializer=tf.initializers.he_normal())
b9 = tf.Variable(tf.compat.v1.random_normal([32]), name='b9')
L9 = tf.matmul(L8,w9) +b9
L9 = tf.compat.v1.layers.batch_normalization(L9) 
L9 = tf.nn.relu(L9)
print(L6) # Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

#L10.
w10 = tf.compat.v1.get_variable('w10', shape = [32, 10], initializer=tf.initializers.he_normal())
b10 = tf.Variable(tf.compat.v1.random_normal([10]), name='b10')
hypothesis = tf.nn.softmax(tf.matmul(L9,w10)+b10)
print(hypothesis) # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

# 3. 컴파일, 훈련

#컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

#훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, train], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print('Epoch :', '%04d'%(epoch + 1), 'cost = {:.9f}'.format(avg_cost))
print('훈련 끗!!!')

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis,1), tf.compat.v1.arg_max(y,1))
accuracy = tf.reduce_mean(tf.compat.v1.cast(prediction, dtype=tf.float32))
print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))