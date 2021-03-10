
# [실습]
#1. sess.run()
#2. InteractiveSession()
#3. eval(session=sess)
# hypothesis를 출력하는 코드를 만들어보자

# print('hypothesis : , ????)

# ================================================================================

import tensorflow as tf
x = [1,2,3]
W = tf.compat.v1.Variable([0.3], tf.float32)
b = tf.compat.v1.Variable([1.0], tf.float32)

hypothesis = W *x + b

# 1번
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(hypothesis)
print("1번째 방법 hypothesis : ", aaa)
sess.close()
# hypothesis :  [1.3       1.6       1.9000001]

# 2번
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = hypothesis.eval() # 변수.eval
print("2번째 방법 hypothesis : ",bbb)
sess.close()
# hypothesis :  [1.3       1.6       1.9000001]

# 3번 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("3번째 방법 hypothesis : ", ccc)
sess.close()
# hypothesis :  [1.3       1.6       1.9000001]


'''
with tf.compat.v1.InteractiveSession().as_default() as sess:
    sess.run(tf.global_variables_initializer())
    print('eval() : ', hypothesis.eval())
'''