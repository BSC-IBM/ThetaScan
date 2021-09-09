# ThetaScan
Behavior-Driven Forecasting for Vertical Auto-Scaling in Container Cloud

## Introduction
Theta-Scan is a method to detect and forecast behavior patterns on CPU/Memory demand in containerized  applications. It  is  based  on  the  deseasonalized  ThetaForecaster algorithm, a time-series method that only requiresthe  observation  window  and  the  expected  periodicity  as parameters  for  a  prediction.  Theta-Scan  is  in  charge  of finding  the  cycle  from  the  observed  window,  indicating stationarity/trending/periodic patterns, and obtaining a forecaster model without additional computational cost.

## Package content
The ThetaScan package contains the necessary code to 1) generate synthetic traces with stationarity/trending/periodic patterns, 2) process the trace and compute resource limit recommendations and 3) visualization.

### Basic elements
* [ThetaScan.py](https://github.com/HiEST/ThetaScan/blob/main/package/ThetaScan/ThetaScan.py): class containing the needed functions to detect behaviours, scan periods and compute ThetaScan recommendations
* [ThetaModel.py](https://github.com/HiEST/ThetaScan/blob/main/package/ThetaScan/ThetaModel.py) : class implementing the Theta Model, the best-fitted deseasonalized and detrended Simplistic Exponential Smoothing (Holt-Winters)
* [utilities.py](https://github.com/HiEST/ThetaScan/blob/main/package/ThetaScan/utilities.py) : support functions to generate synthetic traces, process and visualize them.

### An example
[example.py](https://github.com/HiEST/ThetaScan/blob/main/package/example.py): an example with generation and recommendation using the ThetaScan method

### Sample notebook
[Theta Scan Autoscaling.ipynb](https://github.com/HiEST/ThetaScan/blob/main/examples/Theta%20Scan%20Autoscaling.ipynb): a jupyter notebook with generation and forecast of resources with ThetaScan. Two approaches are included: using a 5-minute fixed window for provisioning resources and a dynamic window dependent on the detected period.

## Using the tool
### Load the package
To use the package import it into your python script:
```from ThetaScan.ThetaScan import *```
The required packages are: ```numpy, scipy, random, matplotlib```

### ThetaScan methods
* ```ThetaScan()```: constructor of the class.
* ```scan(trace)```: scans the trace to detect periods. Returns the detected period and the smape loss for each of the scanned periods.
* ```detect_adf_stationarity(trace)```: performs the Augmented Dickey-Fuller Stationarity test on the trace. Returns true if stationary.
* ```detect_stationarity(trace)```: performs our stationarity test based on the period errors. Returns true if stationary.
* ```detect_trend(trace, period)```: retrieves the trend of the trace (slope, detrended and fitted trend).
* ```detect_behaviour(trace)```: detects the type of behaviour that the trace follows (trending/periodic/stationary).
* ```next_step_forecasting_theta(trace, period, n_steps)```: produces a Theta forecast given the detected period and the steps to produce.
* ```forecast_segment(trace, window_size, i, observation_window, prev_usage)```: detects the behaviour of the trace and produces the next forecast and provisioning limit.
* ```dynamic_forecast_segment(trace, idx_init, idx_end, default_window, observation_window,prev_usage)```: same as previous but with dynamic window.
* ```recommend(trace, window_size, observation_window)```: produces the complete resource recommendation for a resource usage trace.
* ```dynamic_recommend(trace, observation_window)```: same as previous with dynamic time window.

### ThetaModel methods
* ```ThetaModel()```: constructor of the class.
* ```fit(y, sp)```: deseasonalization, detrending and SES fitting.
* ```forecast (n_forecast)```: forecast, resesonalization and retrending.

## References
1. [IEEE CLOUD 2021 Presentation Video](https://s3.amazonaws.com/pf-upload-01/u-59356/0/2021-08-06/8522sjx/CLD_SHT_118-Theta-Scan.mp4)
2. [IEEE CLOUD 2021 Presentation Slides](https://s3.amazonaws.com/pf-upload-01/u-59356/0/2021-08-06/ss12sej/20210904-Cloud2021-video.pdf)
