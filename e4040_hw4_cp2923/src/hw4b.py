"""
Source Code for Homework 4.b of ECBM E4040, Fall 2016, Columbia University

"""
import copy
import os
import timeit
import inspect
import sys
import numpy
import random
from collections import OrderedDict
from sklearn.metrics import f1_score

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from hw4_utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir
from hw4_nn import myMLP, train_nn, RNNSLU
import sys
sys.setrecursionlimit(15000)


def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num, nbit))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    
    return X, Y

#TODO: implement RNN class to learn parity function
class RNN(object):
    def __init__(self, nh, nc, ne, de, cs, normal=True, layernormal = False):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        
        # as many columns as context window size
        # as many lines as words in the sentence
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # labels

        if layernormal == False:
            def recurrence(x_t, h_tm1):
                h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
                s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
                return [h_t, s_t]
            # bundle
            self.params = [self.emb, self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]

        else:
            self.g = theano.shared(name='g',
                                value=numpy.ones(nh,
                                dtype=theano.config.floatX))
            def recurrence(x_t, h_tm1):
                tmp = T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh)
                mean = tmp.mean(0)
                std = tmp.std(0)
                h_t = T.nnet.sigmoid((self.g/std)*(tmp-mean) + self.bh)
                s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
                return [h_t, s_t]
            # bundle
            self.params = [self.emb, self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0, self.g]


        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})
        self.normal = normal

    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        self.sentence_train(words, labels, learning_rate)
        if self.normal:
            self.normalize()

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))
        
    

#TODO: implement LSTM class to learn parity function
class LSTM(object):
    #based on equations in http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """ Elman Neural Net Model Class
    """
    def __init__(self, nh, nc, ne, de, cs, normal=True):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        self.emb = theano.shared(name='embeddings',
                                 value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de))
                                 # add one for padding at the end
                                 .astype(theano.config.floatX))
        self.wx_i = theano.shared(name='wx_i',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wx_f = theano.shared(name='wx_f',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wx_c = theano.shared(name='wx_c',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wx_o = theano.shared(name='wx_o',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de * cs, nh))
                                .astype(theano.config.floatX))
        self.wh_i = theano.shared(name='wh_i',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wh_f = theano.shared(name='wh_f',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wh_c = theano.shared(name='wh_c',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wh_o = theano.shared(name='wh_o',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh_i = theano.shared(name='bh_i',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bh_f = theano.shared(name='bh_f',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bh_c = theano.shared(name='bh_c',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bh_o = theano.shared(name='bh_o',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        self.c0 = theano.shared(name='c0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        
        # bundle
        self.params = [self.emb, self.wx_i, self.wx_f,self.wx_c,self.wx_o,self.wh_i,self.wh_f,self.wh_c,self.wh_o, self.w,
                       self.bh_i,self.bh_f,self.bh_c,self.bh_o, self.b, self.h0,self.c0]

        # as many columns as context window size
        # as many lines as words in the sentence
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y_sentence = T.ivector('y_sentence')  # labels


        def recurrence(x_t, h_tm1,c_tm1):
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wx_f) + T.dot(h_tm1,self.wh_f)+self.bh_f)
             
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wx_i) + T.dot(h_tm1,self.wh_i)+self.bh_i)
            
            
            curve_c_t = T.tanh(T.dot(x_t, self.wx_c) + T.dot(h_tm1,self.wh_c) + self.bh_c)
            
            c_t = i_t*curve_c_t + f_t*c_tm1
            
            o_t = T.nnet.sigmoid(T.dot(x_t,self.wx_o) + T.dot(h_tm1,self.wh_o)+self.bh_o)
            
            h_t = o_t*T.tanh(c_t)
            
            
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t,c_t,s_t]
        
        [h, c, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0,self.c0, None],
                                n_steps=x.shape[0])
                                  
        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})
        self.normal = normal

    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        self.sentence_train(words, labels, learning_rate)
        if self.normal:
            self.normalize()

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))


            
            



#TODO: build and train a MLP to learn parity function
def test_mlp_parity(n_bit,learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=20, n_hidden=500, verbose=False, n_hiddenLayers = 1):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=n_bit,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=2
    )

    # TODO: use your MLP and comment out the classifier object above

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 5  # wait this much longer when a new best is
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

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

            if test_score == 0 and epoch >10:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    return test_score

def genpair(x):
    y = numpy.zeros(x.shape)
    for i in range(x.shape[1]):
        y[:,i] = numpy.mod(numpy.sum(x[:,0:(i+1)], axis=1),2) 
    return y.astype(numpy.int32)

    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(**kwargs):

    """
    Wrapper function for training and testing RNNSLU

    :type fold: int
    :param fold: fold index of the ATIS dataset, from 0 to 4.

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient.

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer.

    :type win: int
    :param win: number of words in the context window.

    :type nhidden: int
    :param n_hidden: number of hidden units.

    :type emb_dimension: int
    :param emb_dimension: dimension of word embedding.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type decay: boolean
    :param decay: decay on the learning rate if improvement stop.

    :type savemodel: boolean
    :param savemodel: save the trained model or not.

    :type normal: boolean
    :param normal: normalize word embeddings after each update or not.

    :type folder: string
    :param folder: path to the folder where results will be stored.

    """
    # process input arguments
    param = {
        'n_bit':8,
        'fold': 3,
        'lr': 0.0970806646812754,
        'verbose': True,
        'decay': True,
        'win': 7,
        'nhidden': 200,
        'seed': 345,
        'emb_dimension': 50,
        'nepochs': 60,
        'savemodel': False,
        'normal': True,
        'folder':'../result',
        'layernormal': False,
        'idx2word': False 
    }
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))
    '''
    # create result folder if not exists
    check_dir(param['folder'])
    '''
    # load the dataset
    print('... loading the dataset')
    
        
    # generate datasets
    train_set = gen_parity_pair(param['n_bit'], 1000)
    valid_set = gen_parity_pair(param['n_bit'], 500)
    test_set  = gen_parity_pair(param['n_bit'], 100)
    
    vocsize = 2
    nclasses = 2
    nsentences = len(train_set[1])
    
    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set
    train_set_y = genpair(train_set_x)
    valid_set_y = genpair(valid_set_x)
    test_set_y = genpair(test_set_x) 
    '''
    train_set_y = train_set_y.astype(numpy.int32)
    valid_set_y = valid_set_y.astype(numpy.int32)
    test_set_y = test_set_y.astype(numpy.int32)
    
    for i in range(3):
        valid_set_y = numpy.column_stack((valid_set_y,valid_set_y))
        train_set_y = numpy.column_stack((train_set_y,train_set_y))
        test_set_y = numpy.column_stack((test_set_y,test_set_y))
    '''
    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    print('... building the model')
    rnn = RNN(
        nh=param['nhidden'],
        nc=nclasses,
        ne=vocsize,
        de=param['emb_dimension'],
        cs=param['win'],
        normal=param['normal'],
        layernormal = param['layernormal']
    )

    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):
        '''
        # shuffle
        shuffle([train_lex, train_ne, train_y], param['seed'])
        '''
        param['ce'] = e
        tic = timeit.default_timer()
        
        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            rnn.train(x, y, param['win'], param['clr'])
            '''
            print('[learning] epoch %i >> %2.2f%%' % (
                e, (i + 1) * 100. / nsentences), ' ')
            print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic), '')
            sys.stdout.flush()
            '''
        
        # evaluation // back into the real world : idx -> words
        predictions_test = numpy.asarray([rnn.classify(numpy.asarray(
                            contextwin(x, param['win'])).astype('int32'))
                            for x in test_set_x])
        predictions_valid = numpy.asarray([rnn.classify(numpy.asarray(
                             contextwin(x, param['win'])).astype('int32'))
                             for x in valid_set_x])
        #return valid_set_y, predictions_valid
        coord = param['n_bit']-1
        res_valid = f1_score(valid_set_y[:, coord], predictions_valid[:, coord])
        res_test = f1_score(test_set_y[:, coord], predictions_test[:, coord])
        '''
        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test,
                             groundtruth_test,
                             words_test,
                             param['folder'] + '/current.test.txt',
                             param['folder'])
        res_valid = conlleval(predictions_valid,
                              groundtruth_valid,
                              words_valid,
                              param['folder'] + '/current.valid.txt',
                              param['folder'])
        '''
        if res_valid > best_f1:

            if param['savemodel']:
                rnn.save(param['folder'])

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid

            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'valid F1', res_valid,
                      'best test F1', res_test)

            param['vf1'], param['tf1'] = res_valid, res_test
            '''
            param['vp'], param['tp'] = res_valid['p'], res_test['p']
            param['vr'], param['tr'] = res_valid['r'], res_test['r']
            '''
            param['be'] = e
            '''
            os.rename(param['folder'] + '/current.test.txt',
                      param['folder'] + '/best.test.txt')
            os.rename(param['folder'] + '/current.valid.txt',
                      param['folder'] + '/best.valid.txt')
            '''
        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'valid F1', param['vf1'],
           'best test F1', param['tf1'],
           'with the model', param['folder'])
#TODO: build and train a LSTM to learn parity function
def test_lstm_parity(**kwargs):
    '''
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    '''

    """
    Wrapper function for training and testing RNNSLU

    :type fold: int
    :param fold: fold index of the ATIS dataset, from 0 to 4.

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient.

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer.

    :type win: int
    :param win: number of words in the context window.

    :type nhidden: int
    :param n_hidden: number of hidden units.

    :type emb_dimension: int
    :param emb_dimension: dimension of word embedding.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type decay: boolean
    :param decay: decay on the learning rate if improvement stop.

    :type savemodel: boolean
    :param savemodel: save the trained model or not.

    :type normal: boolean
    :param normal: normalize word embeddings after each update or not.

    :type folder: string
    :param folder: path to the folder where results will be stored.

    """
    # process input arguments
    param = {
        'n_bit':8,
        'fold': 3,
        'lr': 0.0970806646812754,
        'verbose': True,
        'decay': True,
        'win': 7,
        'nhidden': 200,
        'seed': 345,
        'emb_dimension': 50,
        'nepochs': 60,
        'savemodel': False,
        'normal': True,
        'folder':'../result',
        'layernormal': False,
        'idx2word': False 
    }
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))
    '''
    # create result folder if not exists
    check_dir(param['folder'])
    '''
    # load the dataset
    print('... loading the dataset')
    
        
    # generate datasets
    train_set = gen_parity_pair(param['n_bit'], 1000)
    valid_set = gen_parity_pair(param['n_bit'], 500)
    test_set  = gen_parity_pair(param['n_bit'], 100)
    
    vocsize = 2
    nclasses = 2
    nsentences = len(train_set[1])
    
    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set
    
    train_set_y = genpair(train_set_x)
    valid_set_y = genpair(valid_set_x)
    test_set_y = genpair(test_set_x)
    '''
    for i in range(3):
        valid_set_y = numpy.column_stack((valid_set_y,valid_set_y))
        train_set_y = numpy.column_stack((train_set_y,train_set_y))
        test_set_y = numpy.column_stack((test_set_y,test_set_y))
    '''
    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    print('... building the model')
    rnn = LSTM(
        nh=param['nhidden'],
        nc=nclasses,
        ne=vocsize,
        de=param['emb_dimension'],
        cs=param['win'],
        normal=param['normal']
        )

    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):
        '''
        # shuffle
        shuffle([train_lex, train_ne, train_y], param['seed'])
        '''
        param['ce'] = e
        tic = timeit.default_timer()
        
        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            rnn.train(x, y, param['win'], param['clr'])
            '''
            print('[learning] epoch %i >> %2.2f%%' % (
                e, (i + 1) * 100. / nsentences), ' ')
            print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic), '')
            
            sys.stdout.flush()
            '''
        # evaluation // back into the real world : idx -> words
        predictions_test = numpy.asarray([rnn.classify(numpy.asarray(
                            contextwin(x, param['win'])).astype('int32'))
                            for x in test_set_x])
        predictions_valid = numpy.asarray([rnn.classify(numpy.asarray(
                             contextwin(x, param['win'])).astype('int32'))
                             for x in valid_set_x])
        #return valid_set_y, predictions_valid
        
        coord = param['n_bit'] - 1 
        res_valid = f1_score(valid_set_y[:, coord], predictions_valid[:, coord])
        res_test = f1_score(test_set_y[:, coord], predictions_test[:, coord])
        '''
        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test,
                             groundtruth_test,
                             words_test,
                             param['folder'] + '/current.test.txt',
                             param['folder'])
        res_valid = conlleval(predictions_valid,
                              groundtruth_valid,
                              words_valid,
                              param['folder'] + '/current.valid.txt',
                              param['folder'])
        '''
        if res_valid > best_f1:

            if param['savemodel']:
                rnn.save(param['folder'])

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid

            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'valid F1', res_valid,
                      'best test F1', res_test)

            param['vf1'], param['tf1'] = res_valid, res_test
            '''
            param['vp'], param['tp'] = res_valid['p'], res_test['p']
            param['vr'], param['tr'] = res_valid['r'], res_test['r']
            '''
            param['be'] = e
            '''
            os.rename(param['folder'] + '/current.test.txt',
                      param['folder'] + '/best.test.txt')
            os.rename(param['folder'] + '/current.valid.txt',
                      param['folder'] + '/best.valid.txt')
            '''
        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'valid F1', param['vf1'],
           'best test F1', param['tf1'],
           'with the model', param['folder'])
    
if __name__ == '__main__':
    test_mlp_parity()
