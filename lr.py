import numpy
import theano
import theano.tensor as T
rng = numpy.random

steps=100000
feats=2
x = T.lmatrix("x")
y = T.lvector("y")
w = theano.shared(rng.randn(feats))
b = theano.shared(0.)
print "Modelo inicial:"
print "W (size): " + repr(w.get_value().shape)
print "b (valor): " + repr(b.get_value())

import scipy.io as io
        
print '... cargando datos'
data=io.loadmat('dataLR.mat',squeeze_me=True)
dataIn=data['data'][:,0:2].astype(float)
dataOut = data['data'][:,2].astype(int)

'''N = 400
feats = 2
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
#dataIn=D[0]
#dataOut=D[1]
'''

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial
        
# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)),allow_input_downcast=True)
predict = theano.function(inputs=[x], outputs=prediction,allow_input_downcast=True)

# Train
for i in range(steps):
    pred, err = train(dataIn, dataOut)
    
print "Valores esperados: ", dataOut
print "Valores previstos: ", predict(dataIn)

print "Tasa de acierto: ", map(lambda x,y:x==y, dataOut, predict(dataIn)).count(True)
