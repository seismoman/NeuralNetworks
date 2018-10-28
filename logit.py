import tensorflow as tf
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import timeit

#property of Cooper W. Harris, University of Southern California, April 2018

#get into data directory to read the data
os.chdir('../Outputs/')
# xz = pd.read_csv('../Inputs/z_.time_series.csv', header = None)
#xe = pd.read_csv('../Inputs/e_.time_series.csv', header = None)
# xn = pd.read_csv('../Inputs/n_.time_series.csv', header = None)
# xz = pd.read_csv('../Inputs/z_.spec_coeff.csv', header = None) #spectral coefficients
xe = pd.read_csv('../Inputs/e_.spec_coeff.csv', header = None) #spectral coefficients
# xn= pd.read_csv('../Inputs/n_.spec_coeff.csv', header = None) #spectral coefficients
#X = abs(xz) + abs(xe) + abs(xn)
X = xe
Y = pd.read_csv('../Inputs/1d_labels.csv', header = None)
# Y = pd.read_csv('../Inputs/dummy_labels.csv', header = None)

#dimensionality
depth = 1
xlen = len(X.T)
n_input = xlen
n_hidden = 50

#total number of seismograms being used
traces = len(X)

#parameters
learn_rate = 0.0005
runs = 1

#for plotting
a = []
l = []
s = []
t = []

#for stat tracking
alpha = []
beta = []

#ratio of total data to be used for testing (rest is for training)
divide = int(0.3*traces)

#all data
Xfull = X.as_matrix()

#all features
Yfull = Y.as_matrix()


# Create model
def single_model(X, weights, bias):
    output = tf.matmul(X,weights) + bias
    return output

def double_model(X, weights1, weights2, bias1, bias2):
    layer1 = tf.matmul(X,weights1) + bias1
    layer1 = tf.tanh(layer1)
    output = tf.matmul(layer1,weights2) + bias2
    return output

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

if __name__ == '__main__':

    for k in range(runs):

        start = timeit.default_timer()
        print "run:",k
        #dummy lists
        duma = []
        duml = []
        dums = []

        #randomly split data into training and testing groups (new groups every run)
        index_full = range(1,traces)
        index_test = np.random.random_integers(1, traces-1, divide)
        index_train = list(set(index_full) - set(index_test))
        tr_x = Xfull[index_train]
        tr_y = Yfull[index_train]
        te_x = Xfull[index_test]
        te_y = Yfull[index_test]

        # Variables
        x = tf.placeholder(tf.float32, [None, xlen])
        y_ = tf.placeholder(tf.float32, [None, 1])
        p5 = tf.constant(0.5)  # threshold of Logistic Regression


        if depth == 1:
            weights = tf.Variable(tf.random_normal([xlen,1], mean=0.0, stddev=0.05))
            bias = tf.Variable([0.])
            y_pred = single_model(x,weights,bias)
            y_pred_sigmoid = tf.sigmoid(y_pred)   # for prediction

        if depth == 2:
            weights1 = tf.Variable(tf.random_normal([xlen, n_hidden], mean=0.0, stddev=0.05))
            weights2 = tf.Variable(tf.random_normal([n_hidden, 1], mean=0.0, stddev=0.05))
            bias1 = tf.Variable([0.])
            bias2 = tf.Variable([0.])
            y_pred = double_model(x, weights1, weights2, bias1, bias2)
            y_pred_sigmoid = tf.sigmoid(y_pred)   # for prediction


        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_)
        loss = tf.reduce_mean(x_entropy)
        train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
        delta = tf.abs((y_ - y_pred_sigmoid))
        correct_prediction = tf.cast(tf.less(delta, p5), tf.int32)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Train
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            print('Training...')
            previous = 0
            for i in range(4001):
                batch_xs, batch_ys = tr_x, tr_y
                fd_train = {x: batch_xs, y_: batch_ys.reshape((-1, 1))}
                train_step.run(fd_train)
                dums.append(i)
                duml.append(loss.eval(fd_train))
                duma.append(accuracy.eval(fd_train))

                if i % 500 == 0:
                    new = accuracy.eval(fd_train)
                    if previous >= new:
                        break
                    previous = new
                    loss_step = loss.eval(fd_train)
            train_accuracy = accuracy.eval(fd_train)
            print('  step, loss, accurary = %6d: %8.3f,%8.3f' % (i,
                                            loss_step, train_accuracy))
            s.append(dums)
            l.append(duml)
            a.append(duma)

            # Test trained model
            fd_test = {x: te_x, y_: te_y.reshape((-1, 1))}
            t.append(accuracy.eval(fd_test))
            print('test accuracy = %10.4f' % accuracy.eval(fd_test))

            #grade = 1 if guess is correct, 0 if guess is wrong. does not distinguish false positive from false negative etc
            test_accuracy, grade = sess.run([accuracy,correct_prediction],feed_dict={x: te_x, y_: te_y.reshape((-1, 1))})
            marks = grade - 2*te_y.reshape((-1, 1))
            mlen = len(marks)
            num_b = np.sum(te_y)
            num_e = mlen - num_b
            print mlen, num_b, num_e
            aa = []
            bb = []
            for m in range(mlen):
                if marks[m] == 0:
                    aa.append(1)
                if marks[m] == -2:
                    bb.append(1)

            alpha.append(np.round((float(sum(aa))/num_b),3))
            beta.append(np.round((float(sum(bb))/num_e),3))

            #timing info
            stop = timeit.default_timer()
            mark = stop-start
            mark = '%.3f' % mark
            print "that took", mark, "seconds"


for r in range(runs):
    #plot accuracy
    ax1.plot(s[r],a[r],c='m')
    ax1.set_ylabel('training accuracy')
    ax1.set_xlabel('iterations')
    ax1.scatter(max(s[r]),t[r],marker='x',c='k',s=35)

    #plot loss
    ax2.plot(s[r],l[r],c='b')
    ax2.set_ylabel('training x-entropy')
    ax2.set_xlabel('iterations')

np.savetxt('test_accuracy_new.csv',t,delimiter=',')
np.savetxt('alpha_new.csv',alpha,delimiter=',')
np.savetxt('beta_new.csv',beta,delimiter=',')
plt.show()
##
