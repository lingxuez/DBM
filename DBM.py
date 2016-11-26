## 10-807 HW3
## Lingxue Zhu
## Deep Boltzmann Machine

import cPickle as pickle
import numpy as np
import math, copy, os
from scipy.special import expit

class DBM(object):

	def initModel(self, layerSizes, numChains=10):
		"""
		Initialize model.
		"""
		# self.layerSizes = layerSizes
		# self.numLayers = len(layerSizes)
		# self.numChains = numChains
		self.gibbsChain = self.initGibbsChain(layerSizes, numChains)
		# self.MFmeans = self.initMFmeans(layerSizes)
		self.initParam(layerSizes)

	def initGibbsChain(self, layerSizes, numChains):
		"""
		Initialize K gibbs chains, K given by numChains.
		:param layerSizes: a list containing sizes of each layer
		:param numChains: an integer
		:return chain: a list with length = len(layerSizes)
			such that chain[i] is a K-by-layerSizes[i] nparray
			each row represents one gibbs chain for the i-th layer
		"""
		chain = []
		for size in layerSizes:
			## random (0,1) values in data and hidden layers
			chain += [np.random.uniform(low=0.2, high=0.8, size=(numChains, size))]
		return chain

	def initParam(self, layerSizes):
		"""
		Initialize model parameters, including weights and biases.
		:param layerSizes: a list containing sizes of each layer
		"""
		## one bias term per layer
		self.bias = []
		for size in layerSizes:
			self.bias += [np.zeros((1, size))]

		## one weight matrix between two layers
		self.wt = [np.empty((0, 0))] ## place holder, won't be used
		for i in xrange(1, len(layerSizes)):
			bound = math.sqrt(6.0) / math.sqrt(layerSizes[i-1] + layerSizes[i])
			self.wt += [np.random.uniform(low = -bound, high = bound, 
						size = (layerSizes[i-1], layerSizes[i]))]

	def initMFmeans(self, layerSizes, data=None):
		"""
		Initialize the variational parameters.
		:param layerSizes: a list containing sizes of each layer
		:param data: N-by-D nparray, where D = layerSizes[0]
		:return MFmeans: MFmeans[i] represents the values for the i-th layer, 
			where 0-th equals to data, and others are randomly initialized.
		"""
		if data is None:
			## randomly generated
			data = np.random.uniform(low=0.2, high=0.8, size=(1, layerSizes[0]))

		nsample = data.shape[0]
		MFmeans = [data]
		for i in xrange(1, len(layerSizes)):
			## random mean of hidden layers, not necessarily binary
			MFmeans += [np.random.uniform(low=0.2, high=0.8, size=(nsample, layerSizes[i]))]

		return MFmeans

	def optMFmeans(self, layerSizes, data, steps=10):
		"""
		Compute the optimal variational means for hidden layers given data.
		:param layerSizes: a list containing sizes of each layer
		:param data: N-by-D nparray, where D = layerSizes[0]
		:param steps: the number of fixed-point iterations to perform
		"""
		MFmeans = self.initMFmeans(layerSizes, data)

		for step in xrange(steps):
			## only update hidden layers; data in 0-th layer is unchanged
			for i in xrange(1, len(layerSizes)):
				MFmeans[i] = self.layerCondMean(MFmeans, i)

		return MFmeans

	def layerCondMean(self, values, layer):
		"""
		Compute the comditional mean of the given layer, 
		given values in all layers, using current parameter values.
		:param values: current values in all layers
		:param layer: an integer between [0, len(values)-1]
		:return condMean: the conditional mean of given layer, 
			which has the same shape as values[layer]
		"""
		numLayers = len(self.bias)
		if len(values) != numLayers:
			raise ValueError("values must all layers.")

		if layer < 0 or layer > numLayers-1:
			raise ValueError("invalid layer value.")

		## pre-activate value
		preAct = self.bias[layer]
		## from previous layer (broadcast)
		if layer > 0:
			preAct = preAct + values[layer-1].dot(self.wt[layer])
		## from next layer (broadcast)
		if layer < numLayers-1:
			preAct = preAct + values[layer+1].dot(self.wt[layer+1].transpose())

		## element-wise sigmoid function
		return expit(preAct)

	def updateGibbsChain(self, chain, steps=1):
		"""
		Update the gibbs chain using current model parameter.
		:param chain: a list with length = len(layerSizes)
			such that chain[i] is a K-by-layerSizes[i] nparray
			each row represents one gibbs chain for the i-th layer.
		:param steps: an integer, number of steps
		:return: None. In-place modify chain. 
		"""
		numLayers = len(chain)
		for step in xrange(steps):
			## odd layers first, then even layers
			for i in range(1, numLayers, 2) + range(0, numLayers, 2):
				layerMean = self.layerCondMean(chain, i)
				chain[i] = self.bernoulli(layerMean)

	def bernoulli(self, prob):
		"""
		Element-wise generate binary Bernoulli sample with given probability.
		"""
		f = np.vectorize(lambda p: np.random.binomial(1, p), otypes=[np.float])
		return f(prob)

	def updatParam(self, MFmeans, gibbsChain, rate=0.01):
		"""
		In-place update the weights and biases.
		:param MFmeans: MFmeans[i] represents the values for the i-th layer,
			with size N-by-layerSizes[i], where N is batch size.
		:param gibbsChain: gibbsChain[i] contains K chains for i-th layer,
			with size K-by-layerSizes[i], where each row represents one gibbs chain.
		"""
		if len(MFmeans) != len(gibbsChain):
			raise ValueError("Mis-matched number of layers in MeanField and Gibbs.")

		numLayers = len(MFmeans)
		batch_size, numChains = MFmeans[0].shape[0], gibbsChain[0].shape[0]
		## biases
		for i in xrange(numLayers):
			self.bias[i] += rate * (np.mean(MFmeans[i], axis=0) - 
								np.mean(gibbsChain[i], axis=0))
		## weights
		for i in xrange(1, numLayers):
			self.wt[i] += rate * (MFmeans[i-1].transpose().dot(MFmeans[i]) / batch_size -
						gibbsChain[i-1].transpose().dot(gibbsChain[i]) / numChains)


	def crossEntropyLoss(self, layerSizes, data, steps=2):
		"""
		Compute the cross-entropy reconstruction loss.
		:param data: N-by-layerSizes[0]
		:param steps: an integer
		:return loss: the reconstruction loss
		"""
		numLayers = len(layerSizes)

		## Use mean-field for less noise
		## sampling start from data
		reconChain = self.initMFmeans(layerSizes, data)
		for step in xrange(steps):
			for i in range(1, numLayers, 2) + range(0, numLayers, 2):
				reconChain[i] = self.layerCondMean(reconChain, i)
		
		## reconstruction loss: data versus conditional mean
		reconMean = reconChain[0]
		loss = - np.sum(data * np.log(reconMean) + (1-data) * np.log(1-reconMean))
		loss /= data.shape[0]

		return loss

	def saveModel(self, outfile="DBMmodel", saveModel=True):
		"""
		Save current weights and bias to .npz file.
		"""
		if saveModel:
			pickle.dump((self.wt, self.bias), open(outfile + "_model.pickle", "wb"))
		np.savetxt(outfile+"_W1.csv", self.wt[1], delimiter=",")

	def train(self, hiddenSizes, trainData, valData, batchSize=1,
			numChains=10, rate=0.01, nepoch=10, startEpoch=0, 
			MFstep=10, CDstep=10, evalStep=5,
			outfile="DBMout", startModel="", saveMid=True):
		"""
		Train the DBM using mini-batch stochastic gradient descent.
		:param hiddenSizes: a list with sizes of hidden layers, excluding data layer
		:param trainData: nparray, training data, rows are samples
		:param valData: nparray, validating data, rows are samples
		:param batch_size: an integer, batch size for mini-batch SGD
		:param numChains: an integer, number of persistent chains for training
		:param rate: a float number, learning rate
		:param nepoch: an integer, number of epochs for optimization
		:param MFstep: an integer, number of iterations in mean-field variation inference
		:param evalStep: an integer, number of Gibbs steps for computing loss
		:return: a tuple with training and validating losses in each epoch
		"""

		layerSizes = [trainData.shape[1]] + hiddenSizes
		nTrain, nVal = trainData.shape[0], valData.shape[0]
		trainLoss, valLoss = [], []
		numBatch = int(math.ceil(nTrain / float(batchSize)))
		initRate = rate
		
		## initialize model parameter and persistent chains
		self.initModel(layerSizes, numChains)

		## if startModel is provided, then start from the previous parameters
		if len(startModel) > 0 and os.path.exists(startModel):
			print "Model is trained from", startModel
			(self.wt, self.bias) = pickle.load(open(startModel, "rt"))
		else:
			print "Model is trained from scratch."

		## keep track of the best model
		minValLoss, bestWt, bestBias = None, None, None
		for epoch in xrange(startEpoch, startEpoch+nepoch):
			rate =  max(initRate / (epoch+1), 1e-3)
			## shuffle training samples, mini-batch SGD
			order = np.random.permutation(nTrain)
			for batch in xrange(numBatch):
				data = trainData[batch*batchSize : min((batch+1)*batchSize, nTrain), :]

				self.MFmeans = self.optMFmeans(layerSizes, data, steps=MFstep)
				self.updateGibbsChain(self.gibbsChain, steps=CDstep)
				self.updatParam(self.MFmeans, self.gibbsChain, rate=rate)

			## compute reconstruction losses
			trainLoss += [self.crossEntropyLoss(layerSizes, trainData, evalStep)]
			valLoss += [self.crossEntropyLoss(layerSizes, valData, evalStep)]
			
			## best model
			# if minValLoss is None or valLoss[epoch-startEpoch] < minValLoss:
			# #	print "Updated best model at epoch", epoch
			# 	minValLoss = valLoss[epoch-startEpoch]
			# 	bestWt = copy.deepcopy(self.wt)
			# 	bestBias = copy.deepcopy(self.bias)

			## save the weights per 50 epoch
			if saveMid and epoch % 50 == 0 and epoch < startEpoch+ nepoch-1:
				print "Model saved to", outfile + "_epoch" + str(epoch)
				self.saveModel(outfile + "_epoch" + str(epoch), saveModel=True)

			## track progress
			print "epoch", epoch, "/", startEpoch+nepoch, ", training loss =", trainLoss[epoch-startEpoch], \
					", validating loss =", valLoss[epoch-startEpoch]

		## save the final model
		self.saveModel(outfile + "_epoch" + str(epoch) , saveModel=True)

		## save the best model
		# pickle.dump((bestWt, bestBias), open(outfile + "_epoch" + str(epoch) + "_best.pickle", "wb"))

		return (trainLoss, valLoss)








