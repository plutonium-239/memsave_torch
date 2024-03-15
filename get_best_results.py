import pandas as pd
from glob import glob
from itertools import product


for device,arch in product(['cuda', 'cpu'], ['linear', 'conv']):
	# usage stats
	df = None
	idx_col = ['model', 'case']
	for fname in glob(f'results/usage_stats-{arch}-{device}-*.csv'):
		with open(fname) as f:
			f.readline()
			temp_df = pd.read_csv(f, index_col=idx_col)
		df = temp_df if df is None else pd.concat([df, temp_df])

	if df is not None:
		best_results = df.groupby(idx_col).min()
		# scale
		maxes = best_results.groupby(['model']).max()
		best_results[['Scaled T', 'Scaled M']] = best_results/maxes
		best_results.to_csv('results/best_results-usage_stats.csv')

	# savings
	df = None
	idx_col = ['model', 'input_vjps']
	for fname in glob(f'results/savings-{arch}-{device}*.csv'):
		with open(fname) as f:
			f.readline()
			temp_df = pd.read_csv(f, index_col=idx_col)
		df = temp_df if df is None else pd.concat([df, temp_df])

	if df is not None:
		best_results = df.groupby(idx_col).max()
		best_results.to_csv('results/best_results-savings.csv')
	
