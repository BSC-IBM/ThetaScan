
#-------------------------------------------------------------------------------
# Import AI4DL
#-------------------------------------------------------------------------------

import ThetaScan
from ThetaScan.ThetaScan import ThetaScan

from matplotlib import pyplot as plt
from ThetaScan.utilities import *
from ThetaScan.ThetaModel import *

#-------------------------------------------------------------------------------
# Main Program
#-------------------------------------------------------------------------------

def main():
	
	## 0. Configuration
	N = 1 #number of traces to generate
	ts_len = 2000 # length of time series

	## 1. Generate synthetic dataset
	data = generate_ts_dataset(N, ts_len)
	#trace_train = trace[:int(ts_len*0.75)]

	#th = ThetaModel()
	#fh = np.arange(20) + 1
	#th.fit(trace_train,20)
	#full_trace_pred, y_pred = th.forecast(len(fh))

	TS = ThetaScan()
	for trace in data:
		forecasted_request, forecasted_predicted = TS.dynamic_recommend(trace[-1000:])
		plt.plot(trace[-1000:])
		plt.plot(forecasted_request[-1000:])
		plt.show()



	## 4. Plotting

if __name__ == "__main__":
	main()
