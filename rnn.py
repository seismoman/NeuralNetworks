from __future__ import print_function
import tensorflow as tf
#from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import timeit

#property of Cooper W. Harris, University of Southern California, April 2018

start = timeit.default_timer()

#minimum snr, in feature and data filenames from data prep code
target = 1.0

#get into data directory to read the data
os.chdir('../Outputs/')
# xz = pd.read_csv('../Inputs/z_f.time_series.csv', header = None)
# xe = pd.read_csv('../Inputs/e_f.time_series.csv', header = None)
# xn = pd.read_csv('../Inputs/n_f.time_series.csv', header = None)
xz = pd.read_csv('../Inputs/z_.spec_coeff.csv', header = None) #spectral coefficients
xe = pd.read_csv('../Inputs/e_.spec_coeff.csv', header = None) #spectral coefficients
xn= pd.read_csv('../Inputs/n_.spec_coeff.csv', header = None) #spectral coefficients
X = abs(xz) + abs(xe) + abs(xn)
Y = pd.read_csv('../Inputs/labels.csv', header = None)

#total number of seismograms being used
traces = len(X)

#ratio of total data to be used for testing (rest is for training)
divide = int(0.3*traces)

#all data
Xfull = X.as_matrix()

#all features
Yfull = Y.as_matrix()

#randomly split data into training and testing groups (new groups every run)
index_full = range(1,traces)
index_test = np.random.random_integers(1, traces-1, divide)
index_train = list(set(index_full) - set(index_test))

xtrain = Xfull[index_train]
ytrain = Yfull[index_train]
xtest = Xfull[index_test]
ytest = Yfull[index_test]
btest = np.sum(ytest[:,0])
etest = np.sum(ytest[:,1])

trained = len(xtrain)
tested = len(xtest)

###network parameters###
        ## column vectors: spectral = 129 rows long; time series = 3130 rows long
        ## !!! chunk_size * n_chunks == vector length (# rows)
#num of datapoints per subwindow
chunk_size = 1
#number of subwindows (steps) for each time series
n_chunks = 129
#number of classification classes
n_classes = 2
#number of hidden nuerons
n_hidden = 50
#how many batches to train/test at once
batch_size = traces-divide
#batch_size = int(batch_size*0.5)
#learning rate
learn_rate = 0.0001
#number of runs through all chunks of all data
n_epochs = 350
#number of hidden layers in mrnn
num_layers = 2
#how often to print accuracy & loss to screen
display_step = 10

###noisy expectation maximization parameters###
#add noise yes/no? 1=add nem noise; 0=no nem noise
add_noise = 1
#noise type (type of distribution drawn from to construct noise values)
#options: 0=uniform, 1=gaussian, 2=cauchy (aka: lorentz)
noise_type = 1
#noise variance
noise_var = 0.025
#noise annealing factor (decay)
annfactor = 0.01

###now define functions to call###

#tensorflow Graph input
x = tf.placeholder("float", [None, n_chunks, chunk_size])
y = tf.placeholder("float", [None, n_classes])

#define multilayer recurrent neural network
def RNN(x):

    layer = {'weights':tf.Variable(tf.random_normal([n_hidden,n_classes],stddev=0.15)),
             'biases':tf.Variable(tf.random_normal([n_classes]))}
    x = tf.unstack(x,n_chunks,1)

    #pick type of activation function for hidden layer
    #some choices: tf.tanh, tf.nn.relu, tf.sigmoid, tf.softplus
    activefn = tf.tanh
    lstm_cell = rnn.BasicLSTMCell(n_hidden,activation=activefn,forget_bias=1.0)

    #include dropout
    lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=0.8, output_keep_prob=0.8)

    #make the rnn a multilayer rnn
    lstm_cell = rnn.MultiRNNCell([lstm_cell]*num_layers)

    outputs, states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    output = tf.matmul(outputs[-1],layer['weights'])+layer['biases']

    return output

prediction = RNN(x)

#define results plotter
#plots 2d plot of training error & training accuracy vs epoch
#also plots test performance as a histogram
#saves figure as '${figname}.pdf'
def plot(acc_list,loss_list,step_list,truth,grade,test_accuracy,step):
    figname = "latest_test"
    fig = plt.figure(1)
    plt.cla()
    ax = plt.subplot(2,1,2)
    plt.plot(step_list,loss_list,color='g')
    ax.set_ylabel('loss')
    ax.set_xlabel('iteration')
    ax = plt.subplot(2,1,1)
    plt.plot(step_list,acc_list,color='b')
    plt.scatter(max(step_list),test_accuracy,marker='x',s=100,c='k')
    ax.set_ylabel('accuracy')
    tn = []     #true negative
    tp = []     #true positive
    fn = []     #false negative (beta)
    fp = []     #false positive (alpha)
    for i in range(len(grade)):
        if grade[i] == 0:
            if truth[i] == 0:
                fp.append(1)
            if truth[i] == 1:
                fn.append(1)
        if grade[i] == 1:
            if truth[i] == 1:
                tp.append(1)
            if truth[i] == 0:
                tn.append(1)
    print(str(trained),"trained")
    print(str(btest),"bombs in testing set")
    print(str(etest),"earthquakes in testing set")
    print(str(tested),"tested")
    print(str(len(fn)),"missed nukes")
    print(str(len(tp)),"found nukes")
    print(str(len(fn)+len(tp)),"total nukes")
    print(str(len(fp)),"missed earthquakes")
    print(str(len(tn)),"found earthquakes")
    print(str((len(fp)+len(tn))),"total earthquakes")
    names = ['false negative '+str(len(fn)),'false positive '+str(len(fp)),'true negative '+str(len(tn)),'true positive '+str(len(tp))]
    stop = timeit.default_timer()
    mark = stop-start
    mark = '%.3f' % mark
    perform = float('%.4f' % test_accuracy)
    perform = perform/0.01
    print(str(perform) + '% Accurate, ' + str(mark) + ' seconds')
    print(str(step) + ' epochs')
    print(str(n_hidden) + ' hidden neurons')
    print(str(num_layers) + ' rnn layers')
    fig.set_tight_layout(True)
    os.chdir('../Figures/')
    plt.savefig(figname + 'pdf', format ='pdf')

#define training function that calls on the aforementioned RNN and plotting functions
def train_nn(x):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    init = tf.global_variables_initializer()

    #makes lists of training stats
    loss_list = []
    acc_list = []
    step_list = []

    #start tf.session
    with tf.Session() as sess:
        step = 0
        sess.run(init)

        #keep training until reach max iterations
        while step < n_epochs:
            #reshape inputs
            ind = np.random.random_integers(1, xtrain.shape[0]-1, batch_size)
            train_data = xtrain[ind]
            train_label = ytrain[ind]
            train_data = train_data.reshape((batch_size,n_chunks,chunk_size))
            sess.run(optimizer,feed_dict={x: train_data, y: train_label})

            #Add NEM noise to output
            output_actv = tf.nn.softmax(prediction)
            ay = sess.run(output_actv, feed_dict={x: train_data})
            nv = noise_var/math.pow(step+1, annfactor)

            if noise_type == 0:
                noise = nv*(np.random.uniform(0,1,[batch_size, n_classes]))

            if noise_type == 1:
                noise = nv*(np.random.normal(0,1,[batch_size, n_classes]))

            if noise_type == 2:
                noise = nv*(np.random.standard_cauchy([batch_size, n_classes]))

            #Filter noise to meet increased likelihood condition
            crit1 = noise*np.log(ay+ 1e-6)
            crit = crit1.sum(axis=1)
            index = (crit >= 0)
            noise_crit = np.repeat(index,n_classes)
            noise_index = np.reshape(noise_crit,[batch_size, n_classes])
            nem_noise = noise_index * noise * add_noise
            train_label = train_label + (nem_noise)

            #Train batches with noise in the training output
            #get accuracy
            acc, loss = sess.run([accuracy,cost],feed_dict={x: train_data, y: train_label})
            #get loss
#            loss = sess.run(cost,feed_dict={x: train_data, y: train_label})
            #append accuracy, loss, and step to lists tracking them
            acc_list.append(acc)
            loss_list.append(loss)
            step_list.append(step)
            #update step
            step += 1

            #display performance to screen (plot at end)
            if step%display_step == 0:
                print("Step= " + str(step) + ", Iter= " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.4f}".format(acc))


        #after training is complete (based on epoch), test and plot performances
        test_data = xtest[:divide]
        test_data = test_data.reshape((-1,n_chunks,chunk_size))
        test_label = ytest[:divide]
        test_accuracy,grade = sess.run([accuracy,correct],feed_dict={x: test_data, y: test_label})
        #grade = 1 if guess is correct, 0 if guess is wrong. does not distinguish false positive from false negative etc
        print("Testing Accuracy:" + "{:.4f}".format(test_accuracy))
        truth = test_label[:,0]
        plot(acc_list,loss_list,step_list,truth,grade,test_accuracy,step)


train_nn(x)
plt.ioff()
plt.show()
