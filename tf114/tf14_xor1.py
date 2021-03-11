import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2,1], name='weight'))
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis -y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy 구현

train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32)) 
# cast함수
# 텐서를 새로운 형태로 캐스팅하는데 사용한다.
# 부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다.
# Boolean형태인 경우 True이면 1, False이면 0을 출력한다.


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(501) :
        cost_val, _ = sess.run([cost,train], feed_dict = {x:x_data, y:y_data})

        if step % 200 ==0 :
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data,y:y_data})
    print("Hypothesis(예측값) : ", h, "predict(원래값) : " ,c , "\n Accuarcy : ",a)


#  Accuarcy :  0.75