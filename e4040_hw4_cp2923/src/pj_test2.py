import copy
import os
import timeit
import inspect
import sys
import numpy
import random
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

import sys
from sklearn.preprocessing import normalize
sys.setrecursionlimit(15000)

class RNNSLU3(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, input, nh):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the first hidden layer

        :type cs: int
        :param cs: input lenth

        """
        # parameters of the model
        self.h1 = theano.shared(name='h1', value=numpy.zeros(nh,dtype=theano.config.floatX))
        self.h2 = theano.shared(name='h2', value=numpy.zeros(nh,dtype=theano.config.floatX))
        self.h3 = theano.shared(name='h3', value=numpy.zeros(nh,dtype=theano.config.floatX))

        self.wx1 = theano.shared(name='wx1', value=0.2 * numpy.random.uniform(-1.0, 1.0, (513, nh)).astype(theano.config.floatX))
        self.wx2 = theano.shared(name='wx2', value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
        self.wx3 = theano.shared(name='wx3', value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))

        self.wh1 = theano.shared(name='wh1', value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
        self.wh2 = theano.shared(name='wh2', value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
        self.wh3 = theano.shared(name='wh3', value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))


        # bundle
        self.params = [self.h1, self.h2, self.h3, self.wx1, self.wx2, self.wx3, self.wh1, self.wh2, self.wh3]

        # as many columns as context window size
        # as many lines as words in the sequence
        x = (input.reshape((513,100))).T


        def recurrence(x_t, h1_tm1,h2_tm1,h3_tm1):
            h1_t = T.nnet.relu(T.dot(x_t, self.wx1) + T.dot(h1_tm1, self.wh1))
            h2_t = T.nnet.relu(T.dot(h1_t,self.wx2) + T.dot(h2_tm1, self.wh2))
            h3_t = T.nnet.relu(T.dot(h2_t,self.wx3) + T.dot(h3_tm1, self.wh3))
            return [h1_t, h2_t, h3_t]

        [h1,h2,h3], _ = theano.scan(fn=recurrence,
                               sequences=x,
                               outputs_info=[self.h1, self.h2, self.h3],
                               n_steps=x.shape[0])
        self.output = h3

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.relu):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
def shared_dataset(x):
    shared_x = theano.shared(numpy.asarray(x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    return shared_x
def test_DRNN(datasets, learning_rate=0.1, n_epochs=20, batch_size=5, point = 100):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)
    l = len(datasets)
    z = shared_dataset(datasets[2])
    for i in range(datasets[2].shape[0]):
        datasets[2][i]=normalize(datasets[2][i])
    x = shared_dataset(datasets[2])
    print(x.eval().shape[0])
    y1 = shared_dataset(datasets[0])
    y2 = shared_dataset(datasets[1])
    train_set_x = x[:20]
    test_set_x = x[20:]
    train_set_z = z[:20]
    test_set_z = z[20:]
    train_set_y1 = y1[:20]
    test_set_y1 = y1[20:]
    train_set_y2 = y2[:20]
    test_set_y2 = y2[20:]

    n_train_batches = 20
    n_test_batches = l-20
    # compute number of minibatches for training, validation and testing
#    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
#    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
#    n_valid_batches //= batch_size
    n_test_batches //= batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y1 = T.matrix('y1')  # the labels are presented as 1D vector of
    y2 = T.matrix('y2')
    z = T.matrix('z')                    # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (32, 32) is the size of CIFAR-10 images.
    layer0_input = x

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    # filter_shape: (number of filters, num input feature maps, filter height, filter width)
    layer0 = RNNSLU3(
        input=layer0_input,
        nh = 150
    )
    self.w = theano.shared(name='w', value=0.2 * numpy.random.uniform(-1.0, 1.0, (nh, 1026)).astype(theano.config.floatX))
    self.b = theano.shared(name='b', value=numpy.zeros((1026,), dtype=theano.config.floatX)



    tmp = T.nnet.relu(T.dot(rnn_output, self.w) + self.b)

    tmp1 = tmp[:,:513]
    tmp2 = tmp[:,513:]
    out1 = abs(tmp1)/(abs(tmp1) + abs(tmp2)+10**(-8))*z
    out2 = abs(tmp2)/(abs(tmp1) + abs(tmp2)+10**(-8))*z
    cost = T.mean((out1-y1.reshape((point,513)))**2+(out2-y2.reshape((point,513)))**2)


    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        cost,
        givens={
            x: test_set_x[index],
            y1: test_set_y1[index],
            y2: test_set_y2[index],
            z: test_set_z[index]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = [self.w, self.b] + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    train_model = theano.function(
        [index],
        cost,
        updates = updates,
        givens={
            x: train_set_x[index],
            y1: train_set_y1[index],
            y2: train_set_y2[index],
            z: train_set_z[index]
        }
    )

    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    test_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the test set; in this case we
                                  # check every epoch

    best_test_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = %i' %(iter))
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % test_frequency == 0:

                # compute zero-one loss on test set
                test_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_test_loss = numpy.mean(test_losses)
                print('epoch %i, minibatch %i/%i, test error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_test_loss * 100.))

                # if we got the best test score until now
                if this_test_loss < best_test_loss:

                    #improve patience if loss improvement is good enough
                    if this_test_loss < best_test_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best test score and iteration number
                    best_test_loss = this_test_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best test score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_test_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
