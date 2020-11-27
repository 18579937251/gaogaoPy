import tensorflow as tf

tf.compat.v1.disable_eager_execution()
hello = tf.constant('Hello, TensorFlow!')
sess = tf.compat.v1.Session()
print(sess.run(hello))

import os
from tensorflow.python.client import device_lib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '99'

print(device_lib.list_local_devices())

#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()  #保证sess.run()能够正常运行
#hello = tf.constant('hello,tensorflow')
#sess= tf.compat.v1.Session()    #版本2.0的函数
#print(sess.run(hello))
