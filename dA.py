"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

"""

import time
import sys
import os

import numpy
import scipy.io as io

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


# start-snippet-1
class dA(object):
    """Denoising Auto-Encoder class (dA)
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
    # end-snippet-1

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=15,
            dataset='digits.mat',
            batch_size=20, output_folder='images'):

    print '... cargando datos'
    data = io.loadmat(dataset, squeeze_me=True)
    dataIn = data['X'].astype(float)
    #dataOut = data['data'][:, 2].astype(int)
    train_set_x = theano.shared(numpy.asarray(dataIn,
                    dtype=theano.config.floatX),borrow=True)  
    #train_set_y = T.cast(theano.shared(numpy.asarray(dataOut,
                   # dtype=theano.config.floatX),borrow=True),'int32')

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=20 * 20,
        n_hidden=400
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(20, 20), tile_shape=(20, 20),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=20 * 20,
        n_hidden=400
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.5,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 30% corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % (training_time / 60.))

    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(20, 20), tile_shape=(20, 20),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_50.png')

    os.chdir('../')


if __name__ == '__main__':
    test_dA()
