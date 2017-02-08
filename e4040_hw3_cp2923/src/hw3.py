"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import os
import numpy
import timeit
import theano
import theano.tensor as T
from theano.tensor.signal import pool
import cv2
import matplotlib.pyplot as plt

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn, BNLeNetConvPoolLayer

#Problem 1
#Implement the convolutional neural network architecture depicted in HW3 problem 1
#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py
def test_lenet(datasets, learning_rate=0.1, n_epochs=200, nkerns=[32, 64], batch_size=500, activation = T.tanh):
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


    train_set_x, train_set_y = shared_dataset(datasets[0])
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (32, 32) is the size of CIFAR-10 images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    # filter_shape: (number of filters, num input feature maps, filter height, filter width)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 6 * 6),
    # or (500, 64 * 6 * 6) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6,
        n_out=4096,
        activation=activation
    )
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4096,
        n_out=512,
        activation=activation
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

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
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
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
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
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

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))


#Problem 2.1
#Write a function to add translations
def translate_image(img, left, up, imgplot = False):
    img = numpy.reshape(img,(3,32,32)).transpose(1,2,0)
    nrows,ncols,color = img.shape
    M = numpy.array([[1,0,-left],[0,1,-up]],dtype='f')
    dst = cv2.warpAffine(img,M,(ncols,nrows))
    if imgplot == True:
        plt.imshow(dst)
    return dst.transpose(2,0,1).flatten()
#Implement a convolutional neural network with the translation method for augmentation
def test_lenet_translation(datasets, learning_rate=0.1, n_epochs=200, nkerns=[32, 64], batch_size=500, activation = T.tanh):
    for i in range(len(datasets[0][0])):
        if numpy.random.uniform(0,1) < 0.2:
            datasets[0][0][i] = translate_image(img = datasets[0][0][i], left = 2, up = 2, imgplot = False)

    test_lenet(datasets = datasets, learning_rate=learning_rate, n_epochs=n_epochs, nkerns=nkerns, batch_size=batch_size, activation = activation)

#Problem 2.2
#Write a function to add roatations
def rotate_image(img, imgplot = False):
    angle = numpy.random.uniform(-15,15)
    img = numpy.reshape(img,(3,32,32)).transpose(1,2,0)
    nrows,ncols,color = img.shape
    M = cv2.getRotationMatrix2D((ncols/2,nrows/2),angle,1)
    dst = cv2.warpAffine(img,M,(ncols,nrows))
    if imgplot == True:
        plt.imshow(dst)
    return dst.transpose(2,0,1).flatten()
#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation(datasets, learning_rate=0.1, n_epochs=200, nkerns=[32, 64], batch_size=500, activation = T.tanh):
    for i in range(len(datasets[0][0])):
        if numpy.random.uniform(0,1) < 0.2:
            datasets[0][0][i] = rotate_image(img = datasets[0][0][i], imgplot = False)

    test_lenet(datasets = datasets, learning_rate=learning_rate, n_epochs=n_epochs, nkerns=nkerns, batch_size=batch_size, activation = activation)


#Problem 2.3
#Write a function to flip images
def flip_image(img, imgplot = False):
    img = numpy.reshape(img,(3,32,32)).transpose(1,2,0)
    dst = cv2.flip(img,1)
    if imgplot == True:
        plt.imshow(dst)
    return dst.transpose(2,0,1).flatten()
#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip(datasets, learning_rate=0.1, n_epochs=200, nkerns=[32, 64], batch_size=500, activation = T.tanh):
    for i in range(len(datasets[0][0])):
        if numpy.random.uniform(0,1) < 0.2:
            datasets[0][0][i] = flip_image(img = datasets[0][0][i], imgplot = False)

    test_lenet(datasets = datasets, learning_rate=learning_rate, n_epochs=n_epochs, nkerns=nkerns, batch_size=batch_size, activation = activation)



#Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean
def noise_injection(img, normal = True, imgplot = False):
    if normal == True:
        img = img + numpy.random.normal(0,1,3072)/float(255)
    else:
        img = img + numpy.random.uniform(-1/float(255),1/float(255),3072)
    if imgplot == True:
        plt.imshow(img.reshape(3,32,32).transpose(1,2,0))
    return img

#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input(datasets, learning_rate=0.1, n_epochs=200, nkerns=[32, 64, 128], batch_size=500, activation = T.tanh):
    for i in range(len(datasets[0][0])):
        if numpy.random.uniform(0,1) < 0.2:
            datasets[0][0][i] = noise_injection(img = datasets[0][0][i], normal = True, imgplot = False)

    test_lenet(datasets = datasets, learning_rate=learning_rate, n_epochs=n_epochs, nkerns=nkerns, batch_size=batch_size, activation = activation)

#Problem 3
#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset
def MY_lenet(datasets, learning_rate=0.1, n_epochs=200, nkerns=[32, 64, 128, 256], batch_size=300, activation = T.tanh):
    rng = numpy.random.RandomState(23455)


    train_set_x, train_set_y = shared_dataset(datasets[0])
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (32, 32) is the size of CIFAR-10 images.
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    # filter_shape: (number of filters, num input feature maps, filter height, filter width)
    layer0 = BNLeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize= None
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = BNLeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 32, 32),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize = (2, 2)
    )

    layer2 = BNLeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 16, 16),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=None
    )

    layer3 = BNLeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 16, 16),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(2, 2)
    )


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 6 * 6),
    # or (500, 64 * 6 * 6) = (500, 800) with the default values.
    layer4_input = layer3.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer4 = HiddenLayer(
        rng,
        input=layer4_input,
        n_in=nkerns[3] * 8 * 8,
        n_out=4096,
        activation=activation
    )
    layer5 = HiddenLayer(
        rng,
        input=layer4.output,
        n_in=4096,
        n_out=512,
        activation=activation
    )

    # classify the values of the fully-connected sigmoidal layer
    layer6 = LogisticRegression(input=layer5.output, n_in=512, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer6.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer6.params + layer5.params+layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters

    gparams = [T.grad(cost, param) for param in params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    momentum =theano.shared(numpy.cast[theano.config.floatX](0.6), name='momentum')
    updates = []
    for param in  params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))
    
    
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 10  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            layer0.set_runmode(0)
            layer1.set_runmode(0)
            layer2.set_runmode(0)
            layer3.set_runmode(0)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = %i' %(iter))
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                layer0.set_runmode(1)
                layer1.set_runmode(1)
                layer2.set_runmode(1)
                layer3.set_runmode(1)
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))

def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(numpy.float32(0))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates  

def drop(input, p=0.7):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied

    :type p: float or double between 0. and 1.
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.

    """
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask


#Problem4
#Implement the convolutional neural network depicted in problem4
def MY_CNN(datasets, learning_rate=0.1, n_epochs=100, nkerns=[64, 128, 256], batch_size=300, activation = T.nnet.relu):
    rng = numpy.random.RandomState(23455)

    train_set_x, train_set_y = shared_dataset(datasets[0])
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (32, 32) is the size of CIFAR-10 images.
    
    layer0_input = x.reshape((batch_size, 3, 32, 32))
    layer0_drop = drop(layer0_input,0.7)
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    # filter_shape: (number of filters, num input feature maps, filter height, filter width)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_drop,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=None
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (15-3+1, 15-3+1) = (13, 13)
    # maxpooling reduces this further to (13/2, 13/2) = (6, 6)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 32, 32),
        filter_shape=(nkerns[0], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[0], 16, 16),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=None
    )

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[1], 16, 16),
        filter_shape=(nkerns[1], nkerns[1], 3, 3),
        poolsize=(2, 2)
    )

    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[1], 8, 8),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=None
    )


    layer4_us=T.nnet.abstract_conv.bilinear_upsampling(
        input = layer4.output,
        ratio = 2,
        batch_size = batch_size,
        num_input_channels = nkerns[2]
    )

    layer5 = LeNetConvPoolLayer(
        rng,
        input=layer4_us,
        image_shape=(batch_size, nkerns[2], 16, 16),
        filter_shape=(nkerns[1], nkerns[2], 3, 3),
        poolsize=None
    )

    layer6 = LeNetConvPoolLayer(
        rng,
        input=layer5.output,
        image_shape=(batch_size, nkerns[1], 16, 16),
        filter_shape=(nkerns[1], nkerns[1], 3, 3),
        poolsize=None
    )

    AddedInputs = layer6.output + layer3.forinput

    layer7_input=T.nnet.abstract_conv.bilinear_upsampling(
        input = AddedInputs,
        ratio = 2,
        batch_size = batch_size,
        num_input_channels = nkerns[1]
        )

    layer7 = LeNetConvPoolLayer(
        rng,
        input=layer7_input,
        image_shape=(batch_size, nkerns[1], 32, 32),
        filter_shape=(nkerns[0], nkerns[1], 3, 3),
        poolsize=None
    )

    layer8 = LeNetConvPoolLayer(
        rng,
        input=layer7.output,
        image_shape=(batch_size, nkerns[0], 32, 32),
        filter_shape=(nkerns[0], nkerns[0], 3, 3),
        poolsize=None
    )
    AddedInputs2 = layer8.output + layer1.forinput

    layer9 = LeNetConvPoolLayer(
        rng,
        input=AddedInputs2,
        image_shape=(batch_size, nkerns[0], 32, 32),
        filter_shape=(3, nkerns[0], 3, 3),
        poolsize=None
    )

    # the cost we minimize during training is the NLL of the model
    cost = T.mean((layer9.output - x.reshape((batch_size, 3, 32, 32)))**2)


    # create a function to compute the mistakes that are made by the model

    # create a list of all model parameters to be fit by gradient descent
    params = layer9.params + layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    updates = Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8)
    '''
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
    
    # create a list of gradients for all model parameters
    updates = adadelta(params,grads)
    '''
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    start_time = timeit.default_timer()

    epoch = 0

    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = %i' %(iter))
            cost_ij = train_model(minibatch_index)
        print('epoch %i, MSE = %f' %(epoch,cost_ij))

    end_time = timeit.default_timer()

    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))

    test_model = theano.function(
       [index],
       [layer0_input,layer0_drop,layer9.output],
       givens={
           x: test_set_x[index:index + batch_size]
       }
    )

    return test_model(0)
