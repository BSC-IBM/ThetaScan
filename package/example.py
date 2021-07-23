
#-------------------------------------------------------------------------------
# Import AI4DL
#-------------------------------------------------------------------------------

import ThetaScan
from ThetaScan import ThetaScan

from matplotlib import pyplot as plt
from ThetaScan.timeseries import *
from ThetaScan.ThetaModel import *

#-------------------------------------------------------------------------------
# Main Program
#-------------------------------------------------------------------------------

def main():
	
	## 0. Configuration
	N = 10 #number of traces to generate
	ts_len = 1000 # length of time series

	## 1. Generate synthetic dataset
	ts = generate_ts_dataset(N, ts_len)
	trace = ts[2]
	trace_train = trace[:int(ts_len*0.75)]

	th = ThetaModel()
	th.fit(trace_train,20)
	full_trace_pred, y_pred = th.forecast(int(ts_len*0.25))


	## 3. Autoscaling

	## 4. Plotting

if __name__ == "__main__":
	main()
