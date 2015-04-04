import numpy
import theano
import theano.tensor as T
import scipy.io as io
        
print '... cargando datos'
data=io.loadmat('dataLR.mat',squeeze_me=True)
dataIn=data['data'][:,0:2]
dataOut = data['data'][:,2]

class CapaOculta(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input
       
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX  # @UndefinedVariable
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        #Parametros del modelo
        self.params = [self.W, self.b]
        
    def output(self,x):
        lin_output =  T.dot(x, self.W) + self.b
        return T.tanh(lin_output)


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        

class MLP(object):
    """Clase Perceptron multicapa

    Vamos a definir una sola capa oculta usando la clase CapaOculta que hemos creado anteriormente, y usaremos una capa de
    salida tipo softmax para la que usaremos la clase LogisticRegression.
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):


        # Creamos la capa oculta
        self.hiddenLayer = CapaOculta(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.hiddenLayer2 = CapaOculta(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_hidden,
            activation=T.tanh
        )


        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm: Nos sirve para regularizar.
        self.L1 = (
            abs(self.hiddenLayer.W).sum() + abs(self.hiddenLayer2.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm: otra forma de regularizar.
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum() + (self.hiddenLayer2.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # Return the mean of the negative log-likelihood of the prediction
        # of this model under a given target distribution.
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # Almacenamos los errores
        self.errors = self.logRegressionLayer.errors


        self.params = self.hiddenLayer.params + self.hiddenLayer2.params + self.logRegressionLayer.params
        
        self.output = self.logRegressionLayer.y_pred
        

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=10000,
             batch_size=20, n_hidden=10):
    train_set_x = theano.shared(numpy.asarray(dataIn,
                    dtype=theano.config.floatX),borrow=True)  # @UndefinedVariable
    train_set_y = T.cast(theano.shared(numpy.asarray(dataOut,
                    dtype=theano.config.floatX),borrow=True),'int32')  # @UndefinedVariable

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    print '... building the model'
    
    index = T.lscalar()  # 
    x = T.matrix('x')  # Datos de entrada
    y = T.ivector('y')  # Datos de salida esperados

    rng = numpy.random.RandomState(1234)

    # Construimos el objeto MLP
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=2,
        n_hidden=n_hidden,
        n_out=2
    )
    

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    

    gparams = [T.grad(cost, param) for param in classifier.params]


    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # Entrenamos la red
    print '... entrenando'
    epoch = 0
    while (epoch < n_epochs):
        epoch = epoch + 1
        minibatch_avg_cost = 0
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = minibatch_avg_cost + train_model(minibatch_index)
        print "Epoca: " + repr(epoch) + " - Error medio: " + repr(minibatch_avg_cost/n_train_batches)
      
    # Sacamos los resultados por pantalla
    print dataOut
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.logRegressionLayer.y_pred,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    test = [test_model(i) for i
                                   in xrange(n_train_batches)]

    error_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    test_losses = [error_model(i) for i
                                   in xrange(n_train_batches)]
    
    print test
    print "Error: " + repr(numpy.mean(test_losses))
 
    
if __name__ == '__main__':
    test_mlp()    