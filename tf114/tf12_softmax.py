import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2,1,1],[1,2,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],
          [1,2,5,6],[1,6,6,6],[1,7,6,7]]    # 8,4
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]  # 8,3

x = tf.placeholder(tf.float32, shape = [None,4])
y = tf.placeholder(tf.float32, shape = [None,3])


w = tf.Variable(tf.random_normal([4,3], name='weight'))
b = tf.Variable(tf.random_normal([1,3]), name = 'bias') # y의 출력의 갯수에 맞게 조정한다.(행렬의 덧셈 방법 참고)

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis -y))
loss = tf.reduce_mean(-tf.reduce_sum(y* tf.log(hypothesis), axis=1)) # categorical_crossentropy
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)



with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # 변수 초기화

    for step in range(2001) :
        loss_val, _ = sess.run([loss,train], feed_dict = {x:x_data, y:y_data})

        if step % 200 ==0 :
            print("epochs : ", step, "\n loss : ", loss_val)

    # predict
    predict = sess.run(hypothesis, feed_dict = {x:[[1,11,7,9]]})
    print(predict, sess.run(tf.argmax(predict,1))) # 가장 높은 값에 1을 할당한다.
    # [[0.87936634 0.10488033 0.01575334]] [0]
    
