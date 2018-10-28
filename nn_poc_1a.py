import timeit
start = timeit.default_timer()
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

size = 10000
half = int(size/2)
X = np.random.normal(loc=0,scale=1,size=size)
noise_level = 0.05
E = np.random.normal(loc=0,scale=noise_level,size=size)
Y = X**3 - X**2*np.sin(X) + E
X = np.matrix(X).T
Y = np.matrix(Y).T

#ratio of total data to be used for testing (rest is for training)
divide = int(0.3*size)
index_full = range(1,size)
index_test = np.random.random_integers(1, size-1, divide)
index_train = list(set(index_full) - set(index_test))
x_train = X[index_train]
y_train = Y[index_train]
x_test = X[index_test]
y_test = Y[index_test]

#params
learn_rate = 0.015
epochs = 5001
dims_in = 1
dims_out = 1
n_input = len(x_train.T)
n_hidden = 100

#weights and biases
w1 = tf.Variable(tf.random_normal([n_input,n_hidden], mean=0.0, stddev=0.05))
b1 = tf.Variable([0.])
w2 = tf.Variable(tf.random_normal([n_hidden,n_hidden], mean=0.0, stddev=0.05))
b2 = tf.Variable([0.])
w_out = tf.Variable(tf.random_normal([n_hidden,1], mean=0.0, stddev=0.05))
b_out = tf.Variable([0.])

#actual nn
def nn(x):
    layer1 = tf.add(b1,tf.matmul(x,w1))
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.add(b2,tf.matmul(layer1,w2))
    layer2 = tf.nn.relu(layer2)
    output = tf.add(b_out,tf.matmul(layer2,w_out))
    return output

#tracking lists for plotting
points = [[], []]

#tensorflow setup
xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)
pred = nn(xs)
cost = tf.reduce_mean(tf.square(ys-pred))
init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

#begin
with tf.Session() as sess:
    sess.run(init)

    print "loop, loss"

    #train with training data
    for i in range(1,epochs):
        sess.run([cost,optimizer],feed_dict={xs:x_train,ys:y_train})
        if i % 100 == 0:
            loss = sess.run(cost,feed_dict={xs:x_train,ys:y_train})
            points[0].append(i)
            points[1].append(loss)
            print i, loss
    #test with testing data
    test_error = sess.run(cost, feed_dict={xs:x_test,ys:y_test})
    print 'Test error =', test_error
    model = np.array(sess.run(pred, feed_dict={xs: x_test}))

#plot descent and test score
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(points[0],points[1],'b--')
ax1.scatter(i,test_error,c='g',marker='X',s=50)
ax1.hlines(y=noise_level,xmin=0,xmax=epochs,color='r')
ax2 = fig.add_subplot(212)
real = np.array(y_test)
ax2.scatter(real,model,marker='o',c='k')
ax2.plot([real.min(),real.max()],[real.min(),real.max()],'r--',lw=1.5)


#get timing info
stop = timeit.default_timer()
mark = stop-start
mark = '%.3f' % mark
print "that took", mark, "seconds"

plt.show()
#end
