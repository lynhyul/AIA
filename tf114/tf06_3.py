# tf06_2.py의 lr을 수정해서
# epoch가 2000번보다 적게 만들어라 (2000번 => 2.8890894e-05 [1.0062277] [2.9858427])
# 100번 이하로 만들어봐라
# W = 1.999.... b= 0.999

# [실습]
# palceholder 사용

import tensorflow as tf
tf.set_random_seed(66)  # 이걸 사용하지 않으면 돌릴 때 마다 값이 달라진다.

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])


W = tf.Variable(tf.random_normal([1]), name = 'weight') # weight값 정규분포에 의한 랜덤한 값 한 개를 집어 넣는다.
b = tf.Variable(tf.random_normal([1]), name = 'bias')


hypothesis = x_train * W + b # y=wx+b의 형태

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss = mse

# optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.01) #옵티마이저
train = tf.train.AdamOptimizer(learning_rate=0.4).minimize(cost)

# [실습]
# 1. [4]
# 2. [5, 6]
# 3. [6,7,8]


with tf.Session() as sess : #with문을쓰면 자동으로 close역할도 수행한다.
    sess.run(tf.compat.v1.global_variables_initializer()) #변수 초기화

    for step in range(101) :     # 2001번을 돌리겠다.
        # sess.run을 했던 모든 것들을 반환시키도록 코드를 짠다.
        cost_val,W_val,b_val,_ = sess.run([cost,W,b,train], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        if step %1 ==0:
            print(step, cost_val,W_val,b_val)
    # predict / 갱신된 W와 b 로 hypothesis(model.predict)를 계산
    print('[4] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[4]}))
    print('[5, 6] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[5,6]}))
    print('[6, 7, 8] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[6,7,8]}))
    
## tf06_2 : 2000 2.8890894e-05 [1.0062277] [2.9858427]) /