import scipy.io as io
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_data(start, stop):
	'''
	Cargamos los datos desde matlab y recolocamos los datos listos para pasarselos a pylearn2
	'''
	data=io.loadmat('dataLR.mat',squeeze_me=True)
	
	X = data['data'][:,0:2]
	y = data['data'][:,2]
	yy = np.zeros((X.shape[0],2))
	#y = y.reshape(X.shape[0],1))
	yy[:,0]=1-y
	yy[:,1]=y

	X = X[start:stop,:]
	yy = yy[start:stop,:]
	#y = y[start:stop]

	return DenseDesignMatrix(X=X, y=yy)