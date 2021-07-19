#-------------------------------------------------------------------------------
# General Libraries
#-------------------------------------------------------------------------------

import numpy as np
from scipy import signal

#-------------------------------------------------------------------------------
# Classes
#-------------------------------------------------------------------------------

class ThetaModel:

    def fit (self, dataset, sp, alpha):

        ## 1. Deseasonalize & Detrend
        season = seasonal_decompose(dataset, sp)            	                       ### THIS IS THE SEASON
        deseason = deseasonalize(dataset, season)                                      ### THIS IS THE DESEASONALIZED AND DETRENDED

        ## 2. Obtain Drift (general Trend) for later
        slope, intercept, drift = compute_trend(deseason)                              ### THIS IS THE SLOPE, INTERCEPT AND DRIFT

        ## 3. Obtain Simple Exponential Smoothing (SES)
        fitted, y_next = compute_ses(deseason, alpha)                                  ### THIS IS THE MODEL (Fitted, Next)

        ## Save "Model"
        self.season = season
        self.deseason = deseason
        self.slope = slope
        self.intercept = intercept
        self.drift = drift
        self.fitted = fitted
        self.next = y_next
        self.dataset = dataset
        self.last = len(dataset)

    def forecast (self, n_forecast):
        ## Get new boundaries
        start = self.last
        end = self.last + n_forecast[-1]
        ## 1. Forecast
        y_pred_1 = forecast_ses(self.next, start, end)

        ## 2. Re-Seasonalize
        y_pred_2 = reseasonalize(y_pred_1, self.season, start)

        ## 3. Re-Trend
        y_pred = retrend(y_pred_2, start, end, self.slope, self.intercept)

        ## Join Full Trace
        full_trace_pred = np.concatenate((self.dataset, y_pred))

        return full_trace_pred, y_pred

    def __init__ (self):
        return
