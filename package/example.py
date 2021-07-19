
#-------------------------------------------------------------------------------
# Import AI4DL
#-------------------------------------------------------------------------------

import ThetaScan
from ThetaScan import ThetaScan

from matplotlib import pyplot as plt

#-------------------------------------------------------------------------------
# Main Program
#-------------------------------------------------------------------------------

def main():
	
	## 0. Configuration
	N = 10 #number of traces to generate
	ts_len = 100 # length of time series

	## 1. Generate synthetic dataset
	ts = ThetaScan.ThetaScan()
	dataset = ts.generate_ts_dataset(N, ts_len)
	for trace in dataset:
		plt.plot(trace)
		plt.show()

	## 3. Autoscaling

	## 4. Plotting

if __name__ == "__main__":
	main()
