from ThetaScan.utilities import *
from ThetaScan.ThetaModel import ThetaModel


class ThetaScan():
    '''
    The ThetaScan class: Behavior-Driven Forecasting. This class contains all the methods needed to analyze and forecast usage 
    of resource consumption traces.
    '''
    def __init__(self, default_request=1000, forecaster=ThetaModel(),stat_threshold=0.01):
        '''
        Parameters
        ----------
        default_request (float): starting resource request assigned during the warmup phase
        forecaster (class): forecaster used for fitting the trace
        stat_threshold (float): p-value threshold for stationary test
        '''
        self.default_request = default_request
        self.forecaster = forecaster
        self.stat_threshold = stat_threshold
        self.LIMIT_FACTOR_PREV_USAGE = 5  # limiting constant for avoiding provisioning peaks

    
    
    def scan(self,trace):
        '''
        Scan the given trace to detect the period and get the error for each of the periods
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        
        Returns
        -------
        period (int): detected period
        errors (numpy array): error for each period
        
        '''
        errors = []
        test_size = len(trace) //8 # size of the test set for testing cycles
        max_period = len(trace) //3 # max period to be examined

        y_train, y_test = trace[:-test_size],trace[-test_size:]
        period_values = range(1, max_period)

        for sp_val in period_values:
            self.forecaster.fit(y_train, sp=sp_val)
            y_pred = self.forecaster.forecast(test_size)[1]
            current_error = smape_loss(y_pred, y_test)
            errors.append(current_error)

        period = period_values[np.argmin(errors)]
        return period, errors

    def detect_adf_stationarity(self,trace):
        ''' 
        Augmented Dickey-Fuller test for stationarity.
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        
        Returns
        -------
        stationary (bool): the trace is stationary (true/false)
        
        '''
        adf = sm.tsa.stattools.adfuller(trace)
        stationary = adf[1] < self.stat_threshold
        return stationary

    def detect_stationarity(self,trace):
        ''' 
        ThetaScan stationarity test.
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        
        Returns
        -------
        stationary (bool): the trace is stationary (true/false)
        period_est (int): estimated period
        
        '''
        period_est, error_est = self.scan(trace)
        stationary = np.var(error_est) < self.stat_threshold
        return stationary, period_est

    def detect_trend(self,trace, period):
        ''' 
        ThetaScan trending detection.
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        period (int): detected period of the trace
        
        Returns
        -------
        trace_notrend (numpy array): trace without trending component
        drift (float): drift of the trending component
        slope (float): slope of the trending component
        '''
        self.forecaster.fit(trace, sp=period)
        trace_notrend = trace - self.forecaster.drift
        return (trace_notrend, self.forecaster.drift, self.forecaster.slope)


    def detect_behaviour(self,trace):
        ''' 
        ThetaScan complete behaviour detection.
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        
        Returns
        -------
        c
        scenario (string): behaviour scenario (stationary / constant / probably periodic / periodic)
        '''
        scenarios = ['stat', 'prob.ctan', 'prob.period', 'period']

        ## 1. Stationarity Test
        stationary_1 = self.detect_adf_stationarity(trace)
        stationary_2, period_est = self.detect_stationarity(trace)

        ## 3. Produce recommendation
        if stationary_2:
            if stationary_1:
                scenario = 0  ## ADF = Ours = STAT -> Stationary
            else:
                scenario = 1  ## ADF = NonStat / Ours = STAT -> Most of them CTAN (full or partial)
        else:
            if stationary_1:
                scenario = 2  ## ADF = Stat / Ours = NonStat -> Time-Dependant or Periodic with low period (ADF false positiva)
            else:
                scenario = 3  ## ADF = NonStat / Ours = NonStat -> Proceed to Predict, Periodic
        y_notrend, y_pred, slope = self.detect_trend(trace,period_est)

        return period_est, scenarios[scenario]


    def next_step_forecasting_theta(self,trace, period, n_steps):
        ''' 
        Forecast the number of indicated steps given the period.
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        period (int): period
        n_steps (int): number of next steps to predict
        
        
        Returns
        -------
        y_pred (numpy array): prediction for the next steps
        '''
        self.forecaster.fit(trace, period)
        y_pred = self.forecaster.forecast(n_steps)[1]
        return y_pred

    def forecast_segment(self, trace, window_size, i, observation_window, prev_usage):
        ''' 
        Forecast the next provisioning segment with fixed time window.
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        window_size (int): fixed window size for next request
        i (int): index for the current provisioning window 
        observation_window (int): sight of the ThetaScan, number of past values that are used to compute recommendations
        prev_usage (numpy array): usage in the previous segment
        
        
        Returns
        -------
        prediction (numpy array): request for the next time window based on forecast
        forecast (numpy array): granular forecast
        '''
        
        observed_segment = min(i * window_size, observation_window)  # choose the minimum number of steps between the maximum observation window and the available number of time steps
        observed_segment_index = range(i * window_size - observed_segment, i * window_size)

        period_est, scenario = self.detect_behaviour(trace[observed_segment_index])
        if (scenario == "period" or scenario == "prob.period"):
            history = period_est
        else:
            history = window_size

        if history > (i * window_size // 2):  # this only happens in the first 10 minutes maximum (warm-up)
            prediction = np.percentile(prev_usage, 95)  # our prediction in the warm-up is the 95 percentile of the seen trace
            forecast = prev_usage
        else:
            if (scenario == "period" or scenario == "prob.period"):
                forecast = self.next_step_forecasting_theta(trace[observed_segment_index], period_est, max(period_est, window_size))
                prediction = max(forecast)

            else:
                forecast = self.next_step_forecasting_theta(trace[observed_segment_index], window_size, window_size)
                prediction = max(forecast)

        limit = max(prev_usage) * self.LIMIT_FACTOR_PREV_USAGE
        if prediction > limit:
            prediction = limit
            forecast = np.clip(forecast, None, limit)

        return prediction, forecast[:window_size]

    def dynamic_forecast_segment(self, trace, idx_init, idx_end, default_window, observation_window,prev_usage):
        ''' 
        Forecast the next provisioning segment with dynamically changing window.
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        idx_init (int): starting granular index for the current time step
        idx_end (int): last granular index for the last time step that we are going to predict
        default_window (int): fixed default window, when there is no periodicity
        observation_window (int): sight of the ThetaScan, number of past values that are used to compute recommendations
        prev_usage (numpy array): usage in the previous segment
        
        
        Returns
        -------
        prediction (numpy array): request for the next time window based on forecast
        forecast (numpy array): granular forecast
        history (int): size of the dynamic window
        '''
        observed_segment = min(idx_end, observation_window)  # choose the minimum number of steps between the maximum observation window and the available number of time steps
        observed_segment_index = range(idx_end - observed_segment, idx_end)

        period_est, scenario = self.detect_behaviour(trace[observed_segment_index])

        if (scenario == "period" or scenario == "prob.period"):
            history = period_est
        else:
            history = default_window

        if (history > (idx_end// 2)):  # this only happens in the first 10 minutes maximum (warm-up)
            prediction = np.percentile(prev_usage,
                                       95)  # our prediction in the warm-up is the 95 percentile of the seen trace
            forecast = np.repeat(prediction,history)
        else:
            if (scenario == "period" or scenario == "prob.period"):
                forecast = self.next_step_forecasting_theta(trace[observed_segment_index], period_est,period_est)
                prediction = max(forecast)
            else:
                forecast = self.next_step_forecasting_theta(trace[observed_segment_index], default_window, default_window)
                prediction = max(forecast)
        return prediction, forecast, history


    def recommend(self, trace, window_size=20, observation_window=(4 * 60) * 2):
        ''' 
        Compute a limit recommendation based on the resource usage trace
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        window_size (int): provisioning window size
        observation_window (int): sight of the ThetaScan, number of past values that are used to compute recommendations
        
        
        Returns
        -------
        forecasted_request (numpy array): granular forecast of resource usage
        forecasted_predicted (numpy array): requested usage based on the forecast
        '''
        trace_len = len(trace)
        forecasted_request = np.zeros(trace_len)
        forecasted_predicted = np.zeros(trace_len)
        forecasted_request[:window_size] = self.default_request
        forecasted_predicted[:window_size] = self.default_request
        steps = trace_len // window_size

        for i in range(1, steps):
            pre_step_idxs = range((i - 1) * window_size, i * window_size)  # range of previous indexes (granular)
            previous_usage = trace[pre_step_idxs]
            if i * window_size + window_size < trace_len:
                cur_step_idxs = range(i * window_size, (i + 1) * window_size)  # current steps to provision
            else:  # if we arrive to the end of the trace
                cur_step_idxs = range(i * window_size, trace_len)
            forecasted_request[cur_step_idxs], forecasted_predicted[cur_step_idxs] = self.forecast_segment(trace, window_size, i,observation_window,previous_usage)
        return forecasted_request, forecasted_predicted

    def dynamic_recommend(self, trace, observation_window=(4 * 60) * 2):
        ''' 
        Compute a limit recommendation based on the resource usage trace
        
        Parameters
        ----------
        trace (numpy array): resource consumption trace
        observation_window (int): sight of the ThetaScan, number of past values that are used to compute recommendations
        
        
        Returns
        -------
        forecasted_request (numpy array): granular forecast of resource usage
        forecasted_predicted (numpy array): requested usage based on the forecast
        '''
        default_window =20
        trace_len = len(trace)
        forecasted_request = np.zeros(trace_len)
        forecasted_predicted = np.zeros(trace_len)
        forecasted_request[:default_window] = self.default_request
        forecasted_predicted[:default_window] = self.default_request

        idx_init = 0
        idx_end = default_window

        while(idx_end<trace_len):
            pre_step_idxs = range(idx_init, idx_end)  # range of previous indexes (granular)
            previous_usage = trace[pre_step_idxs]

            dynamic_request, dynamic_forecast, period = self.dynamic_forecast_segment(trace, idx_init, idx_end, default_window,observation_window,previous_usage)
            if idx_end + period < trace_len:
                cur_step_idxs = range(idx_end, idx_end+period)  # current steps to provision
                idx_init =cur_step_idxs[0]
                idx_end = cur_step_idxs[-1]
            else:  # if we arrive to the end of the trace
                cur_step_idxs = range(idx_end, trace_len)
                idx_init =cur_step_idxs[0]
                idx_end = cur_step_idxs[-1]

            forecasted_request[cur_step_idxs], forecasted_predicted[cur_step_idxs] = dynamic_request,dynamic_forecast[:(idx_end-idx_init)+1]
            if idx_init == idx_end: break
        return forecasted_request, forecasted_predicted


    #
