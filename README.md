# Deep Boltzmann Machine

A python implementation of training Deep Boltzmann Machine and generating Gibbs samples.

## Default hyper-parameters

To train a DBM:

```
sh run.sh
```

After training, obtain Gibbs samples from the trained model:

```
sh run_gibbs.sh
```

## Other hyper-parameters

To train a DBM with `H1` units in the first hidden layer, 
and `H2` units the second hidden layer, 
using initial learning rate `r`, batch size `N`,
and train the model for `T` epochs with `K` persistent chains, run:

```
python trainDBM.py --trainData <path/to/train/data> --valData <path/to/val/data> \
  --hiddenSizes <H1> <H2> --nepoch <T> --rate <r> --batchSize <N> --numChains <K> \
  --outfile <output_filename> --outdir <output_directory>
```

To obtain $M$ Gibbs samples, each with $K$ steps, run the following command,
where you need to specify the path to the `.pickle` file 
(the trained model given by `trainDBM.py`), 
and the path to save the Gibbs samples in a `.csv` file:

```
python gibbsDBM.py --steps <K> --chains <M> \
  --model <path/to/input/pickle/file> --outfile <path/to/output/csv/file>
```
