import tensorflow as tf
tf.set_random_seed(66)  # 이걸 사용하지 않으면 돌릴 때 마다 값이 달라진다.

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(0.9, name = 'weight')
b = tf.Variable(0.3, name = 'bias')

# sess = tf.Session()
# sess.run(tf.compat.v1.global_variables_initializer()) #변수 초기화
# print(sess.run(W), sess.run(b)) # [0.06524777] [1.4264158]

hypothesis = x_train * W + b # y=wx+b의 형태

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss = mse

optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.01) #옵티마이저
train = optimizer.minimize(cost)

# # sess 선언 및 변수 초기화
# sess = tf.Session()
# sess.run(tf.compat.v1.global_variables_initializer()) #변수 초기화

# for step in range(100) :                   # 2001번을 돌리겠다. / 
#     sess.run(train)
#      if step %  ==20:
#        print(step,sess.run(cost), sess.run(W), sess.run(b))
# sess.close() # 메모리 사용을 닫아준다.

    

# sess 선언 및 변수 초기화
with tf.Session() as sess : #with문을쓰면 자동으로 close역할도 수행한다.
    sess.run(tf.compat.v1.global_variables_initializer()) #변수 초기화

    for step in range(100) :                   # 2001번을 돌리겠다.
        sess.run(train)
        if step%20 ==0:
            print(step,sess.run(cost), sess.run(W), sess.run(b))



