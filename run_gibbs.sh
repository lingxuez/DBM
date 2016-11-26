

model="out/DBM_h100_model.pickle" ## path to trained DBM model (.pickle file)
outfile="out_gibbs/DBM_gibbs_samples.csv" ## path to output file (.csv file)
steps=1000 ## number of Gibbs steps
chains=100 ## number of Gibbs chains to sample

python gibbsDBM.py -m $model -s $steps -c $chains -o $outfile 
