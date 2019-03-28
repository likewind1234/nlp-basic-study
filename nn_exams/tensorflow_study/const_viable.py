from __future__ import print_function,division
import tensorflow as tf
from basic_python.basictools import pause
import numpy as np

#build a graph
print("build a graph")
a=tf.constant([[1,2],[3,4]])
b=tf.constant([[1,1],[0,1]])
print("a:",a)
print("b:",b)
print("type of a:",type(a))
c=tf.matmul(a,b)
print("c:",c)
print("\n")
#construct a 'Session' to excute the graph
sess=tf.Session()

# Execute the graph and store the value that `c` represents in `result`.
print("excuted in Session")
result_a=sess.run(a)
result_a2=a.eval(session=sess)
print("result_a:\n",result_a)
print("result_a2:\n",result_a2)

result_b=sess.run(b)
print("result_b:\n",result_b)

result_c=sess.run(c)
print("result_c:\n",result_c)

with tf.variable_scope("space1",reuse=False):
    v1=tf.get_variable(name="V1",shape=(2,2),dtype=tf.float32,initializer=tf.initializers.ones())
    print("name of v1:",v1.name)

    v2 = tf.get_variable(name="V2", shape=(2, 2), dtype=tf.float32, initializer=tf.initializers.zeros())
    print("name of v2:", v2.name)

with tf.variable_scope("space2",reuse=False):
    v3=tf.get_variable(name="V1",shape=(2,2),dtype=tf.float32,initializer=tf.initializers.ones())
    print("name of v3:",v3.name)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(v1))
    print(sess.run(v2))
    print(sess.run(v3))


pause()
#create a Variable
w=tf.Variable(initial_value=[[1,2],[3,4]],dtype=tf.float32)
x=tf.Variable(initial_value=[[1,1],[1,1]],dtype=tf.float32)
y=tf.matmul(w,x)
z=tf.sigmoid(y)
print(z)
init_op=tf.global_variables_initializer()


with tf.Session() as session:
    session.run(init_op)
    z=session.run(z)
    print(z)


pause(True)
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  #print(sess.run(y))  # ERROR: will fail because x was not fed.
  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.