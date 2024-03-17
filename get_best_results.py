import pandas as pd
from glob import glob
from itertools import product

case_mapping = {
	'None': 'All',
	'grad_input + no_grad_conv_weights + no_grad_conv_bias + no_grad_linear_weights + no_grad_linear_bias + no_grad_norm_weights + no_grad_norm_bias': 'Input',
	'no_grad_linear_weights + no_grad_linear_bias + no_grad_norm_weights + no_grad_norm_bias': 'Conv',
	'no_grad_conv_weights + no_grad_conv_bias + no_grad_linear_weights + no_grad_linear_bias': 'Norm'
}

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
		df = df.rename(index=case_mapping, level=1)
		df['Memory Usage (GB)'] = df['Memory Usage (MB)']/1024
		df = df.drop(columns=['Memory Usage (MB)'])
		best_results = df.groupby(idx_col).min()
		# scale
		maxes = best_results.groupby(['model']).max()
		best_results[['Scaled T', 'Scaled M']] = best_results/maxes
		best_results.to_csv(f'results/best_results-{arch}-{device}-usage_stats.csv')

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
		best_results.to_csv(f'results/best_results-{arch}-{device}-savings.csv')
	
