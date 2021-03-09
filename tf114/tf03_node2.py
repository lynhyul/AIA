# [실습]
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈
# 나머지
# 맹그러라!!!!

import tensorflow as tf


sess = tf.Session()

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# 덧셈
node3 = tf.add(node1,node2)
print('sess.run(node3) : ', sess.run(node3)) # sess.run(node3) :  5.0


# 뺄셈
node3 = tf.subtract(node1,node2)
print('sess.run(node3) : ', sess.run(node3)) # sess.run(node3) :  -1.0 


# 곱셈
node3 = tf.multiply(node1,node2)
print('sess.run(node3) : ', sess.run(node3)) # sess.run(node3) :  6.0


# 나눗셈
node3 = tf.divide(node1,node2)
print('sess.run(node3) : ', sess.run(node3)) # sess.run(node3) :  0.6666667

# 나머지
node3 = tf.math.mod(node1,node2)
print('sess.run(node3) : ', sess.run(node3)) #sess.run(node3) :  2.0