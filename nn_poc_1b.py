import timeit
start = timeit.default_timer()
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#inputs
size = 10000
X0 = np.random.normal(loc=0,scale=1,size=size)   #input data
X1 = np.random.normal(loc=0,scale=1,size=size)
noise_level = 0.05
E = np.random.normal(loc=0,scale=noise_level,size=size)    #Gaussian noise
#X = np.matrix(X0)   #make vector input a matrix for tensorflow
X = np.vstack((X0,X1)) #put all input data into one matrix
X = np.matrix(X).T  #take transpose for tensorflow
print np.shape(X)

##define output function y=f(x0,...)+noise
#Y = X0**3 - X0**2*np.sin(X0) + E
Y = X0**2 + X1**3 + E
#Y = np.exp(X1)**2 - X1**2 + np.sin(X1) + E
Y = np.matrix(Y).T      #get transpose for tensorflow


#prepare input output
divide = int(0.5*size) #ratio of total data to be used for testing (rest is for training)
index_full = range(1,size)
index_test = np.random.random_integers(1, size-1, divide)
index_train = list(set(index_full) - set(index_test))
x_train = X[index_train]    #training output
y_train = Y[index_train]    #training output
x_test = X[index_test]  #testing input
y_test = Y[index_test]  #testing output

'''
    mx = 1, my = 1, h = 2, lr = 0.015, mh1 = 100, mh2 = 100, relu
    mx = 2, my = 1, h = 3, lr = 0.005, mh1 = 100, mh2 = 100, mh3 = 100, relu
'''

#params
h = 3   #number of hidden layers (matrices between input & output)
learn_rate = 0.005   #controls how quickly weights are updated during backpropagation
epochs = 5001   #how many runs
n_input = len(x_train.T)    #number of datapoints
mh1 = 100    #number of neurons in hidden layer 1
mh2 = 100   #number of neurons in hidden layer 2
mh3 = 100   #number of neurons in hidden layer 3

#choose desired activation function
def active(this):
    that = tf.nn.relu(this)     #seems to perform best
#    that = tf.sigmoid(this)
#    that = tf.tanh(this)
    return that

#actual nn
def l2nn(x):
    layer1 = tf.add(b1,tf.matmul(x,w1)) #multiply data by weights, add bias
    layer1 = active(layer1) #normalize layer output with activation function
    layer2 = tf.add(b2,tf.matmul(layer1,w2))    #multiply data by weights, add bias
    layer2 = active(layer2) #normalize layer output with activation function
    output = tf.add(b_out,tf.matmul(layer2,w_out))  #multiply data by weights, add bias
    return output

def l3nn(x):
    layer1 = tf.add(b1,tf.matmul(x,w1)) #multiply data by weights, add bias
    layer1 = tf.nn.relu(layer1) #normalize layer output with activation function
    layer2 = tf.add(b2,tf.matmul(layer1,w2))    #multiply data by weights, add bias
    layer2 = tf.nn.relu(layer2) #normalize layer output with activation function
    layer3 = tf.add(b3,tf.matmul(layer2,w3))    #multiply data by weights, add bias
    layer3 = tf.nn.relu(layer3) #normalize layer output with activation function
    output = tf.add(b_out,tf.matmul(layer3,w_out))  #multiply data by weights, add bias
    return output

#tracking lists for plotting
points = [[], []]

#sets up dynamic memory in tensorflow for backpropagation
xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)

#choose proper architecture for given network depth
if h == 2:
    #weights and biases
    w1 = tf.Variable(tf.random_normal([n_input,mh1], mean=0.0, stddev=0.05))
    b1 = tf.Variable([0.])
    w2 = tf.Variable(tf.random_normal([mh1,mh2], mean=0.0, stddev=0.05))
    b2 = tf.Variable([0.])
    w_out = tf.Variable(tf.random_normal([mh2,1], mean=0.0, stddev=0.05))
    b_out = tf.Variable([0.])
    pred = l2nn(xs)     #neural net output
if h == 3:
    #weights and biases
    w1 = tf.Variable(tf.random_normal([n_input,mh1], mean=0.0, stddev=0.05))
    b1 = tf.Variable([0.])
    w2 = tf.Variable(tf.random_normal([mh1,mh2], mean=0.0, stddev=0.05))
    b2 = tf.Variable([0.])
    w3 = tf.Variable(tf.random_normal([mh2,mh3], mean=0.0, stddev=0.05))
    b3 = tf.Variable([0.])
    w_out = tf.Variable(tf.random_normal([mh3,1], mean=0.0, stddev=0.05))
    b_out = tf.Variable([0.])
    pred = l3nn(xs)     #neural net output

#set up error function, solver (optimizer), and initialize with random variables
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
ax1.scatter(i,test_error,c='g',marker='X',s=75)
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
