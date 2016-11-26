## 10-807 hw3
## Lingxue Zhu
## Train a DBM model.

from DBM import *
import os, argparse

def get_data(filename, dim_feature, cutoff=0.5):
	"""
	Load data from a file and convert to binary using the given cutoff,
	and only keep the first dim_feature columns.
	(for MNIST data, the last column is label and should be ignored)
	"""
	data = np.loadtxt(filename, dtype=float, delimiter=",")
	data_labels = data[:, dim_feature]
	data = data[:, xrange(dim_feature)]
	## convert to binary
	data = np.greater_equal(data, cutoff).astype(int)
	return (data_labels, data)


#####################
## train DBM
#####################
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-hid', '--hiddenSizes', dest='hiddenSizes', nargs="+", type=int, 
							default=[100, 100])
	parser.add_argument('-r', '--rate', dest='rate', type=float, default=0.001)
	parser.add_argument('-train', '--trainData', dest='train', type=str, default="data/digitstrain.txt")
	parser.add_argument('-val', '--valData', dest='val', type=str, default="data/digitsvalid.txt")
	parser.add_argument('--batchSize', dest='batchSize', type=int, default=10)
	parser.add_argument('--nepoch', dest='nepoch', type=int, default=30)
	parser.add_argument('--MFstep', dest='MFstep', type=int, default=10)
	parser.add_argument('--CDstep', dest='CDstep', type=int, default=1)
	parser.add_argument('--evalStep', dest='evalStep', type=int, default=10)
	parser.add_argument('--numChains', dest='numChains', type=int, default=10)
	parser.add_argument('--dim_feature', dest='dim_feature', type=int, default=784)
	parser.add_argument('--outfile', dest='outfile', type=str, default="DBM")
	parser.add_argument('--outdir', dest='outdir', type=str, default="out")
	parser.add_argument('--startEpoch', dest='startEpoch', type=int, default=0)
	parser.add_argument('--startModel', dest='startModel', type=str, default="")
	params = vars(parser.parse_args())

	outdir = params["outdir"]
	hiddenSizes = params["hiddenSizes"]
	print "Training DBM using K=", params["numChains"]

	## load data and convert to binary
	(trainLabel, trainData) = get_data(params["train"], dim_feature=params["dim_feature"])
	(valLabel, valData) = get_data(params["val"], dim_feature=params["dim_feature"])
	
	## output file
	if not os.path.exists(outdir):
		os.mkdir(outdir)
	prefix = outdir + "/" + params["outfile"] #+ "_h" + str(hiddenSizes[0]) + "_r" + str(params["rate"])
	
	## DBM
	myDBM = DBM()
	(train_loss, val_loss) = myDBM.train(hiddenSizes=hiddenSizes, 
		trainData=trainData, valData=valData, batchSize=params["batchSize"],
		numChains=params["numChains"], rate=params["rate"], nepoch=params["nepoch"], 
		MFstep=params["MFstep"], CDstep=params["CDstep"], evalStep=params["evalStep"],
		outfile=prefix, startModel=params["startModel"], startEpoch=params["startEpoch"],
		saveMid=False)

	## save loss	
	np.savetxt(prefix + "_epoch" + \
			str(params["startEpoch"]) + "-" + str(params["nepoch"] + params["startEpoch"]) +\
			"_loss.csv",
		np.column_stack((np.array(train_loss), np.array(val_loss))), 
		delimiter=",")
	print "output written to ", prefix
