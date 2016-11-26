
hid=100 ## number of hidden units in each layer
nepoch=200 ## number of epochs to run
outdir="out/" ## output directory
outfile="DBM_h100" ## prefix of output filename
train="data/digitstrain.txt" ## path to training data
valid="data/digitsvalid.txt" ## path to validating data

python trainDBM.py -train $train -val $valid -hid $hid $hid \
	--nepoch $nepoch --outfile $outfile --outdir $outdir
