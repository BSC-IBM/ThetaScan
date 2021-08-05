# ThetaScan
Behavior-Driven Forecastingfor Vertical Auto-Scaling in Container Cloud framework

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
