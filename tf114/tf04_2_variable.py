import tensorflow as tf

sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name= 'test')

init = tf.global_variables_initializer()

sess.run(init) # 변수 초기화

print(sess.run(x)) # [2.]