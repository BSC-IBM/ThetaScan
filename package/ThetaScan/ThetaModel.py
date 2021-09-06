from ThetaScan.utilities import *

class ThetaModel:
    ''' Custom implementation of the Theta Model.
    '''
    def __init__(self, alpha = 0.2):
        '''
        Parameters
        ----------
        alpha (float): constant for the SES forecast
        '''
        self.alpha = alpha


    def fit (self, y, sp):
        ''' 
        Fit a time series to the Theta Model

        Parameters
        ----------
        y (numpy array): time series
        sp (int): period
        '''

        ## 1. Deseasonalize & Detrend
        season = seasonal_decompose(y, sp)            	                       ### THIS IS THE SEASON
        deseason = deseasonalize(y, season)                                      ### THIS IS THE DESEASONALIZED AND DETRENDED

        ## 2. Obtain Drift (general Trend) for later
        slope, intercept, drift = compute_trend(deseason)                              ### THIS IS THE SLOPE, INTERCEPT AND DRIFT

        ## 3. Obtain Simple Exponential Smoothing (SES)
        fitted, y_next = compute_ses(deseason, self.alpha)                                  ### THIS IS THE MODEL (Fitted, Next)

        ## Save "Model"
        self.season = season
        self.deseason = deseason
        self.slope = slope
        self.intercept = intercept
        self.drift = drift
        self.fitted = fitted
        self.next = y_next
        self.dataset = y
        self.last = len(y)

    def forecast (self, n_forecast):
        ''' 
        Forecast a number of steps based on fitted curve.

        Parameters
        ----------
        full_trace_pred (numpy array): complete trace with prediction (past + forecasted future)
        y_pred (numpy array): prediction for the next time steps

        '''
    
        ## Get new boundaries
        start = self.last
        end = self.last + n_forecast
        ## 1. Forecast
        y_pred_1 = forecast_ses(self.next, start, end)

        ## 2. Re-Seasonalize
        y_pred_2 = reseasonalize(y_pred_1, self.season, start)

        ## 3. Re-Trend
        y_pred = retrend(y_pred_2, start, end, self.slope, self.intercept)

        ## Join Full Trace
        full_trace_pred = np.concatenate((self.dataset, y_pred))

        return full_trace_pred, y_pred