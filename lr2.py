import numpy
import theano
import theano.tensor as T
from theano import sandbox, Out

theano.config.floatX='float32'

rng = numpy.random

import scipy.io as io
        
print '... cargando datos'
'''data=io.loadmat('dataLR.mat',squeeze_me=True)
dataIn=data['data'][:,0:2].astype(theano.config.floatX)
dataOut = data['data'][:,2].astype(theano.config.floatX)'''
training_steps = 10000

N = 40000
feats = 784
D = (rng.randn(N, feats).astype(theano.config.floatX),
rng.randint(size=N, low=0, high=2).astype(theano.config.floatX))
dataIn=D[0]
dataOut=D[1]

# Declare Theano symbolic variables
x = theano.shared(dataIn, name="x")
y = theano.shared(dataOut, name="y")
w = theano.shared(rng.randn(dataIn.shape[1]).astype(theano.config.floatX), name="w")
b = theano.shared(numpy.asarray(0., dtype=theano.config.floatX), name="b")
x.tag.test_value = dataIn
y.tag.test_value = dataOut
#print "Initial model:"
#print w.get_value(), b.get_value()

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b)) # Probability of having a one
prediction = p_1 > 0.5 # The prediction that is done: 0 or 1
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1) # Cross-entropy
cost = xent.mean() + 0.01*(w**2).sum() # The cost to optimize
gw,gb = T.grad(cost, [w,b])

# Compile expressions to functions
train = theano.function(
            inputs=[],
            outputs=[prediction, xent],
            updates={w:w-0.01*gw, b:b-0.01*gb},
            name = "train",allow_input_downcast=True)
predict = theano.function(inputs=[], outputs=prediction,
            name = "predict",allow_input_downcast=True)

if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
        train.maker.fgraph.toposort()]):
    print 'Used the cpu'
elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
          train.maker.fgraph.toposort()]):
    print 'Used the gpu'
else:
    print 'ERROR, not able to tell if theano used the cpu or the gpu'
    print train.maker.fgraph.toposort()

for i in range(training_steps):
    pred, err = train()
#print "Final model:"
#print w.get_value(), b.get_value()

print "target values for D"
print dataOut

print "prediction on D"
print predict()
