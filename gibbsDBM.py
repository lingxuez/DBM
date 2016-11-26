## Generate a Gibbs chain using trained DBM

from DBM import *
import numpy as np
import argparse
import cPickle as pickle

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", dest="model", type=str)
	parser.add_argument("-s", "--steps", dest="steps", type=int, default=1000)
	parser.add_argument("-c", "--chains", dest="chains", type=int, default=100)
	parser.add_argument("-o", "--outfile", dest="outfile", type=str, default="gibbs")
	params = vars(parser.parse_args())
	
	myDBM = DBM()
	## trained model
	(myDBM.wt, myDBM.bias) = pickle.load(open(params["model"], "r"))
	
	## layer sizes
	layerSizes = []
	for i in xrange(len(myDBM.bias)):
		layerSizes += [myDBM.bias[i].shape[1]]

	## gibbs chain
	chain = myDBM.initGibbsChain(layerSizes, params["chains"])	
	myDBM.updateGibbsChain(chain, steps=params["steps"])

	## save first layer of chain
	np.savetxt(params["outfile"], chain[0], delimiter=",")

