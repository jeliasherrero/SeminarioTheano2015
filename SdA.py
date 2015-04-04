import os
import sys
import time
import scipy.io as io

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import CapaOculta, LogisticRegression
from dA import dA


class SdA_MLP_Layer(object):
		"""Creamos una clase para definir una capa consistente en un MLP y un dA.
		Esta capa podra ser unida junto con otras para formar stacked
		auto-encoders layers.
		"""
		def __init__(
				self,
				input,
				numpy_rng,
				theano_rng=None,
				n_ins=784,
				hidden_layers_size=500,
				corruption_level=0.1
		):

				if not theano_rng:
						theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
				# allocate symbolic variables for the data
				self.x = T.matrix('x')  # the data is presented as rasterized images

				layer_input = self.x

				self.sigmoid_layer = CapaOculta(rng=numpy_rng,
																				input=layer_input,
																				n_in=n_ins,
																				n_out=hidden_layers_size,
																				activation=T.nnet.sigmoid)


				self.dA_layer = dA(numpy_rng=numpy_rng,
													 theano_rng=theano_rng,
													 input=layer_input,
													 n_visible=n_ins,
													 n_hidden=hidden_layers_size,
													 W=self.sigmoid_layer.W,
													 bhid=self.sigmoid_layer.b)

				self.params = self.sigmoid_layer.params

				self.output = self.sigmoid_layer.output

		def pretraining(self, train_set_x, batch_size):
				# index to a [mini]batch
				index = T.lscalar('index')  # index to a minibatch
				corruption_level = T.scalar('corruption')  # % of corruption to use
				learning_rate = T.scalar('lr')  # learning rate to use
				# begining of a batch, given `index`
				batch_begin = index * batch_size
				# ending of a batch given `index`
				batch_end = batch_begin + batch_size

				# get the cost and the updates list
				cost, updates = self.dA_layer.get_cost_updates(corruption_level, learning_rate)
				# compile the theano function
				fn = theano.function(
						inputs=[
								index,
								theano.Param(corruption_level, default=0.2),
								theano.Param(learning_rate, default=0.1)
						],
						outputs=cost,
						updates=updates,
						givens={
								self.x: train_set_x[batch_begin: batch_end]
						}
				)
				return fn

		def getOut(self, input, num_inputs):
				index = T.lscalar('index')
				test_model = theano.function(
		        inputs=[index],
		        outputs=self.sigmoid_layer.output,
		        givens={
		            self.x: input[index * num_inputs: (index + 1) * num_inputs]
		        })

				return test_model(0)

class SdA_MLP_network(object):
		def __init__(self,
								hidden_layers_sizes=[500, 500],
									corruption_levels=[0.1, 0.1],
									n_outs=10,
									n_ins=400):

				numpy_rng = numpy.random.RandomState(89677)

				self.params = []
				self.sigmoid_layers = []

				self.x = T.matrix('x')
				self.y = T.ivector('y')

				self.layer0 = SdA_MLP_Layer(
						input=self.x,
						numpy_rng=numpy_rng,
						n_ins=n_ins,
						hidden_layers_size=hidden_layers_sizes[0],
						corruption_level=corruption_levels[0])

				self.layer1 = SdA_MLP_Layer(
						input=self.layer0.output,
						numpy_rng=numpy_rng,
						n_ins=hidden_layers_sizes[0],
						hidden_layers_size=hidden_layers_sizes[1],
						corruption_level=corruption_levels[1])

				self.logLayer = LogisticRegression(
					input=self.layer1.sigmoid_layer.output,
					n_in=hidden_layers_sizes[-1],
					n_out=n_outs)

				#self.params = self.layer0.params + self.layer1.params + self.logLayer.params
				self.sigmoid_layers.append(self.layer0)
				self.sigmoid_layers.append(self.layer1)
				self.sigmoid_layers.append(self.logLayer)

				self.params.extend(self.layer0.params)
				self.params.extend(self.layer1.params)
				self.params.extend(self.logLayer.params)
				self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
				self.errors = self.logLayer.errors(self.y)


		def pretraining(self, input, batch_size,
										corruption_levels, pretraining_epochs,
										pretrain_lr):

				
				n_train_batches = input.get_value(borrow=True).shape[0]
				n_train_batches /= batch_size

				fn0 = self.layer0.pretraining(train_set_x=input,
																			batch_size=batch_size)
				for epoch in xrange(pretraining_epochs):
					c = []
					for batch_index in xrange(n_train_batches):
						c.append(fn0(index=batch_index,
												 corruption=corruption_levels[0],
												 lr=pretrain_lr))
					print 'Pre-training layer 0, epoch %d, cost ' % (epoch),
					print numpy.mean(c)

				#print input.get_value()
				output = self.layer0.getOut(input, input.get_value().shape[0])

				out = theano.shared(numpy.asarray(output,
										dtype=theano.config.floatX),borrow=True)
				#print out.shape[0]
				#print input.shape[0]

				fn1 = self.layer1.pretraining(train_set_x=out,
																								batch_size=batch_size)

				for epoch in xrange(pretraining_epochs):
					c = []
					for batch_index in xrange(n_train_batches):
						c.append(fn1(index=batch_index,
												 corruption=corruption_levels[1],
												 lr=pretrain_lr))
					print 'Pre-training layer 1, epoch %d, cost ' % (epoch),
					print numpy.mean(c)

		def finetune(self, 
									train_set_x, train_set_y,
									batch_size, learning_rate):
			n_train_batches = train_set_x.get_value(borrow=True).shape[0]
			n_train_batches /= batch_size
			index = T.iscalar('index')  # index to a [mini]batch
			gparams = T.grad(self.finetune_cost, self.params)
			updates = [(param, param - gparam * learning_rate)
				for param, gparam in zip(self.params, gparams)]

			print index.dtype
			print self.y.dtype
			print  train_set_y
			train_fn = theano.function(
						inputs=[index],
						outputs=self.finetune_cost,
						updates=updates,
						givens={
								self.x: train_set_x[
										index * batch_size: (index + 1) * batch_size
								],
								self.y: train_set_y[
										index * batch_size: (index + 1) * batch_size
								]
						},
						name='train'
				)

			test_score_i = theano.function(
						[index],
						self.errors,
						givens={
								self.x: train_set_x[
										index * batch_size: (index + 1) * batch_size
								],
								self.y: train_set_y[
										index * batch_size: (index + 1) * batch_size
								]
						},
						name='test'
				)

				# Create a function that scans the entire test set
			def test_score():
						return [test_score_i(i) for i in xrange(n_test_batches)]

			return train_fn, test_score

		def training(self, train_set_x, train_set_y, batch_size, learning_rate):
			# Entrenamos la re
			print '... entrenando'

			n_train_batches = train_set_x.get_value(borrow=True).shape[0]
			n_train_batches /= batch_size

			train_fn, test_model = self.finetune(train_set_x, train_set_y,
									batch_size, learning_rate)
			epoch = 0
			while (epoch < n_epochs):
				epoch = epoch + 1
				minibatch_avg_cost = 0
				for minibatch_index in xrange(n_train_batches):
					minibatch_avg_cost = minibatch_avg_cost + train_fn(minibatch_index)

def test_SdA(finetune_lr=0.1, pretraining_epochs=1, corruption_levels=[.1, .2],
						 pretrain_lr=0.001, training_epochs=1000,
						 dataset='digits.mat', batch_size=1):
		print '... cargando datos'
		data = io.loadmat(dataset, squeeze_me=True)
		# Datos de entrenamiento
		
		dataIn = data['X'].astype(float)
		dataOut = data['y'].astype(int)
		train_set_x = theano.shared(numpy.asarray(dataIn,
										dtype=theano.config.floatX),borrow=True)
		train_set_y = T.cast(theano.shared(numpy.asarray(dataOut,
										dtype=theano.config.floatX),borrow=True),'int32')
		
		sda = SdA_MLP_network(
				n_outs=10,
				n_ins=20 * 20
		)

		sda.pretraining(train_set_x, batch_size,
			corruption_levels, pretraining_epochs, pretrain_lr)

		sda.training(train_set_x, train_set_y, batch_size, finetune_lr)

if __name__ == '__main__':
    test_SdA()		