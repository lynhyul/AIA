# 즉시 실행 모드
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf



print(tf.executing_eagerly())   # False(1. 버전) // True(2. 버전)
 
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   # False


print(tf.__version__)   # 1.14.0 / 2.4.0

hello = tf.constant("Hello World")

print(hello)
# # Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session() # 1버전에서는 session을 따로 선언해줘야 프린트 출력이 가능
print(sess.run(hello))
# # b'Hello World'
