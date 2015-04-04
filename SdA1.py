import scipy.io as io

import numpy
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import CapaOculta, LogisticRegression
from dA import dA


class SdA_MLP_Layer(object):
	def __init__(self, input, numpy_rng, theano_rng=None, n_ins=784, hidden_layers_size=500, corruption_level=.1):

		# Tensor de entrada
		self.input = input

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		# Capa sigmoidea
		self.sigmoid_layer = CapaOculta(rng=numpy_rng, input=input, n_in=n_ins, n_out=hidden_layers_size, activation=T.nnet.sigmoid)

		# Capa Denoising Autoencoder
		self.dA_layer = dA(numpy_rng=numpy_rng, theano_rng=theano_rng, input=input, n_visible=n_ins, n_hidden=hidden_layers_size, W=self.sigmoid_layer.W, bhid=self.sigmoid_layer.b)

		self.params = self.sigmoid_layer.params

		self.output = self.sigmoid_layer.output

	def pretraining(self, train_input, batch_size):
		# Creamos los tensores
		corruption_level = T.scalar('corruption')
		learning_rate = T.scalar('lr')
		index = T.iscalar('index')

		# Calculamos los indices del lote
		batch_begin = index * batch_size
		batch_end = batch_begin + batch_size

		# Calculamos los costes y las actualizaciones
		cost, updates = self.dA_layer.get_cost_updates(corruption_level, learning_rate)

		fn = theano.function(
			inputs=[
				index,
				theano.Param(corruption_level, default=0.2),
				theano.Param(learning_rate, default=0.1)],
			outputs=cost,
			updates=updates,
			givens={self.input: train_input[batch_begin:batch_end]})
		return fn

	def getOut(self, inputData, num_inputs):
		index = T.iscalar('index')
		batch_begin = index * num_inputs
		batch_end = (index+1) * num_inputs

		n_train_batches = inputData.get_value(borrow=True).shape[0] / batch_size

		salida = theano.function(
			inputs=[index],
			outputs=self.output,
			givens={self.input: inputData[batch_begin:batch_end]})
		return salida(0)


class SdA_MLP_network(object):
	def __init__(self, input, output, hidden_layers_sizes=[500, 500], corruption_levels=[0.1, 0.1], n_outs=10, n_ins=400):
		
		numpy_rng = numpy.random.RandomState(89677)

		self.input = input
		self.output = output

		self.corruption_levels = corruption_levels

		self.layer0 = SdA_MLP_Layer(
			input=input,
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

		self.log_layer = LogisticRegression(
			input=self.layer1.output,
			n_in=hidden_layers_sizes[-1],
			n_out=n_outs)

		self.params = self.log_layer.params + self.layer0.params + self.layer1.params

		self.finetune_cost = self.log_layer.negative_log_likelihood(output)
		self.errors = self.log_layer.errors(output)

	def pretraining(self, inputData, batch_size, pretraining_epochs, pretrain_lr):

		n_train_batches = inputData.get_value(borrow=True).shape[0] / batch_size
		
		fn0 = self.layer0.pretraining(inputData, batch_size)

		for epoch in xrange(pretraining_epochs):
			c = []
			for batch_index in xrange(n_train_batches):
				c.append(fn0(
					index=batch_index,
					corruption=self.corruption_levels[0],
					lr=pretrain_lr))
			print 'Pre-training capa 0, epoch %d, cost %f' % (epoch, numpy.mean(c))

		output = self.layer0.getOut(inputData, inputData.get_value().shape[0])

		out = theano.shared(numpy.asarray(output, dtype=theano.config.floatX), borrow=True)

		fn1 = self.layer1.pretraining(out, batch_size)

		for epoch in xrange(pretraining_epochs):
			c = []
			for batch_index in xrange(n_train_batches):
				c.append(fn1(
					index=batch_index,
					corruption=self.corruption_levels[1],
					lr=pretrain_lr))
			print 'Pre-training capa 1, epoch %d, cost %f' % (epoch, numpy.mean(c))

	def fine_tune(self, inputData, outputData, batch_size, learning_rate):
		# Indice neceario
		index = T.iscalar('index')

		gparams = T.grad(self.finetune_cost, self.params)
		updates = [(param, param - gparam * learning_rate) for param, gparam in zip(self.params, gparams)]

		train_fn = theano.function(
			inputs=[index],
			outputs=self.finetune_cost,
			updates=updates,
			givens={
				self.input: inputData[index * batch_size:(index + 1) * batch_size],
				self.output: outputData[index * batch_size:(index + 1) * batch_size]}, name='train')

		'''test_fn = theano.function(
			inputs=[index],
			outputs=self.errors,
			updates=updates,
			givens={
				self.input: inputData[index * batch_size:(index + 1) * batch_size],
				self.output: outputData[index * batch_size:(index + 1) * batch_size]}, name='train')

		def test_score():
			return [test_fn(i) for i in xrange(self.n_train_batches)]'''

		return train_fn

	def training(self, inputData, outputData, batch_size, n_epochs, learning_rate):
		print "...entrenando"

		n_train_batches = inputData.get_value(borrow=True).shape[0] / batch_size

		train_fn = self.fine_tune(inputData, outputData, batch_size, learning_rate)

		epoch = 0
		coste = numpy.zeros((n_epochs, 1))
		epoca = []
		coste = []
		while (epoch < n_epochs):
			epoch = epoch + 1
			minibatch_avg_cost = 0
			for minibatch_index in xrange(n_train_batches):
				minibatch_avg_cost = minibatch_avg_cost + train_fn(minibatch_index)
			print 'Training MLP, epoch %d, cost %f' % (epoch, minibatch_avg_cost/n_train_batches)
			coste.append(minibatch_avg_cost/n_train_batches)
			epoca.append(epoch)
		plt.plot(epoca, coste)
		plt.show()

	def test(self, inputData, outputData, batch_size):
		index = T.iscalar('index')
		n_train_batches = inputData.get_value(borrow=True).shape[0] / batch_size

		test_fn = theano.function(
			inputs=[index],
			outputs=self.errors,
			givens={
				self.input: inputData[index * batch_size:(index + 1) * batch_size],
				self.output: outputData[index * batch_size:(index + 1) * batch_size]}, name='test')

		def test_score():
			return [test_fn(i) for i in xrange(n_train_batches)]
		print 1. - numpy.mean(test_score())

finetune_lr = 0.1
pretraining_epochs = 500
corruption_levels = [.1, .2]
pretrain_lr = 0.001
training_epochs = 1000
dataset = 'digits.mat'
batch_size = 500

print "...cargando datos"
data = io.loadmat(dataset, squeeze_me=True)

dataIn = data['X'].astype(float)
dataOut = data['y'].astype(int)

for i in range(dataOut.shape[0]):
	if (dataOut[i] == 10):
		dataOut[i] = 0

train_set_x = theano.shared(numpy.asarray(dataIn, dtype=theano.config.floatX), borrow=True)
train_set_y = T.cast(theano.shared(numpy.asarray(dataOut, dtype=theano.config.floatX), borrow=True), 'int32')

n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size

x = T.matrix('x')  # Datos de entrada
y = T.ivector('y')	# Salida esperada

sda = SdA_MLP_network(
	x,
	y,
	n_outs=10,
	n_ins=20 * 20)


sda.pretraining(train_set_x, batch_size, pretraining_epochs, pretrain_lr)

sda.training(train_set_x, train_set_y, batch_size, training_epochs, finetune_lr)

sda.test(train_set_x, train_set_y, batch_size)
