import scipy.io as io
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import random as rd
from random import randrange
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd

class matlabData(DenseDesignMatrix):
    def __init__(self):
        self.class_names = ['0', '1']
        data=io.loadmat('digits.mat',squeeze_me=True)
        
        X = data['X']
        y = data['yp']
        
        xx=np.zeros((X.shape))
        yy=np.zeros((y.shape))
        for i in range((X.shape[0])):
            rd_index=randrange(0,X.shape[0])
            xx[i]=X[rd_index]
            yy[i]=y[rd_index]

        super(matlabData, self).__init__(X=xx, y=yy)
 
ds = matlabData()

first_layer = mlp.ConvRectifiedLinear(layer_name='conv1', output_channels=64, irange= .05, 
                                      kernel_shape= [5, 5],pool_shape= [4, 4],
                                      pool_stride= [2, 2],max_kernel_norm= 1.9365)

second_layer = mlp.ConvRectifiedLinear(layer_name='conv2', output_channels=64, irange= .05, 
                                      kernel_shape= [5, 5],pool_shape= [4, 4],
                                      pool_stride= [2, 2],max_kernel_norm= 1.9365)

output_layer = mlp.Softmax(max_col_norm= 1.9365,
                     layer_name= 'output',
                     n_classes= 10,
                     istdev= .05)

from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.cost import MethodCost
from pylearn2.costs.mlp import WeightDecay
from pylearn2.termination_criteria import And
from pylearn2.termination_criteria import EpochCounter

trainer = sgd.SGD(learning_rate=.01, batch_size=100, 
                  learning_rule=Momentum(init_momentum=0.5),
                  cost=SumOfCosts(costs=[
                        MethodCost(method='cost_from_X'),
                        WeightDecay(coeffs=[.00005, .00005, .00005])
                  ]),
                  termination_criterion=And(criteria=[
                        EpochCounter(max_epochs=500)
                  ]))

from pylearn2.space import Conv2DSpace

entrada=Conv2DSpace(shape=[20,20],num_channels=1)
layers = [first_layer,second_layer,output_layer]
ann = mlp.MLP(layers=layers,batch_size=100,input_space=entrada)
trainer.setup(ann, ds)