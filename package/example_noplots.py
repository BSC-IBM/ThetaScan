from ThetaScan.ThetaScan import *

def main():
	
	## 0. Configuration
	N = 4 #number of traces to generate
	ts_len = 1000 # length of time series

	## 1. Generate synthetic dataset
	data, names = generate_ts_dataset(N, ts_len)

	## 2. Recommend
	TS = ThetaScan()
	for trace in data:
		forecasted_request, forecasted_predicted = TS.dynamic_recommend(trace)

if __name__ == "__main__":
	main()
