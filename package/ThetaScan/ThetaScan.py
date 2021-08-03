# -------------------------------------------------------------------------------
# General Libraries
# -------------------------------------------------------------------------------
from matplotlib import pyplot as plt
# -------------------------------------------------------------------------------
# Other libraries and tools
# -------------------------------------------------------------------------------
from ThetaScan.utilities import *
from ThetaScan.ThetaModel import ThetaModel


class ThetaScan():
    def __init__(self, default_request=1000, forecaster=ThetaModel(),stat_threshold=0.01):
        self.forecaster = forecaster
        self.default_request = default_request
        self.stat_threshold = stat_threshold
        self.LIMIT_FACTOR_PREV_USAGE = 5  # we can change this constant, we use it to avoid forecasting peaks


    def scan(self,trace,test_size,sp_max):
        filterwarnings('ignore')
        errors = []

        if test_size >= len(trace):
            test_size = len(trace) // 2

        y_train, y_test = temporal_train_test_split(trace, test_size=test_size)
        sp_values = list(range(1, min(sp_max, len(y_train) // 2)))

        if not isinstance(y_train, pd.DataFrame):
            y_train = pd.Series(y_train)

        if not isinstance(y_test, pd.DataFrame):
            y_test = pd.Series(y_test, index=range(len(y_train), len(y_train) + len(y_test)))

        for sp_val in sp_values:
            fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
            self.forecaster.fit(y_train, sp=sp_val)
            y_pred = self.forecaster.forecast(len(fh))[1]
            y_pred = pd.Series(y_pred, index=y_test.index)
            current_error = smape_loss(y_pred, y_test)
            errors.append(current_error)

        period = sp_values[np.argmin(errors)]
        period_error = pd.Series(errors)
        period_error.name = "Period error"
        return (period, period_error)

    def detect_adf_stationarity(self,trace):
        adf = sm.tsa.stattools.adfuller(trace)
        stationary = adf[1] < self.stat_threshold
        return stationary

    def detect_stationarity(self,trace,test_size,sp_max):
        period_est, error_est = self.scan(trace,test_size,sp_max)
        return np.var(error_est) < self.stat_threshold, period_est

    def detect_trend(self,trace):
        if not isinstance(trace, pd.DataFrame):
            trace = pd.Series(trace)

        forecaster = PolynomialTrendForecaster(degree=1)
        transformer = Detrender(forecaster=forecaster)

        fh_ins = -np.arange(len(trace))  # in-sample forecasting horizon
        trace_notrend = transformer.fit_transform(trace)
        trace_pred = forecaster.fit(trace).predict(fh=fh_ins)

        trace_notrend = pd.Series(trace_notrend, name="No Trend")
        trace_pred = pd.Series(trace_pred, name="Fitted Trend")

        slope = (trace_pred[trace_pred.index[-1]] - trace_pred[trace_pred.index[0]]) / len(trace_pred)

        return (trace_notrend, trace_pred, slope)


    def detect_behaviour(self,trace,limit,test_size,sp_max):
        scenarios = ['stat', 'prob.ctan', 'prob.period', 'period']

        ## 1. Stationarity Test
        stationary_1 = self.detect_adf_stationarity(trace)
        stationary_2, period_est = self.detect_stationarity(trace,test_size,sp_max)

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
        y_notrend, y_pred, slope = self.detect_trend(trace)

        if period_est < limit:
            scenario = 0

        ## 4. Return all flags/estimations
        information = (
        stationary_1, stationary_2, scenarios[scenario], np.mean(trace), np.std(trace), np.max(trace), period_est,
        slope)

        return pd.DataFrame([information],
                            columns=['adf.stat', 'our.stat', 'scenario', 'mean', 'st.dev', 'max', 'period',
                                     'slope'])

    def next_step_forecasting_theta(self,trace, step):
        if not isinstance(trace, pd.DataFrame):
            trace = pd.Series(trace)
        if step < 1:
            y_pred = [0]
        else:
            #forecaster = ThetaModel()
            self.forecaster.fit(trace, step)
            fh = np.arange(step) + 1  ## <- LENGTH WILL BE HIST (fit) + NEXT STEP (predict)
            y_pred = self.forecaster.forecast(len(fh))[1]  ## <- PREDICT STEP
        return y_pred

    def forecast_segment(self, trace, window_size, i,observation_window,prev_usage):
        observed_segment = min(i * window_size, observation_window)  # choose the minimum number of steps between the maximum observation window and the available number of time steps
        observed_segment_index = range(i * window_size - observed_segment, i * window_size)
        step_test = observed_segment // 8  # the test size for testing cycles
        step_max = observed_segment // 3  # max cycle to be examined
        decision_thetascan = self.detect_behaviour(trace[observed_segment_index],window_size,step_test,step_max)

        if (decision_thetascan["scenario"][0] == "period" or decision_thetascan["scenario"][0] == "prob.period"):
            history = (decision_thetascan["period"].to_numpy())[0]  # Detected periodicity, get period as history for forecasting
        else:
            history = window_size #observed_segment  # or window_size

        if (history > (i * window_size // 2)):  # this only happens in the first 10 minutes maximum (warm-up)
            seen_trace = trace[(i * window_size - history):i * window_size]
            prediction = np.percentile(seen_trace,
                                       95)  # our prediction in the warm-up is the 95 percentile of the seen trace
            forecast = seen_trace[:window_size]  # our forecast in the warm-up is the seen trace
        else:
            forecast = self.next_step_forecasting_theta(trace[:i * window_size],
                                                   history)  # based on the seen trace, predict the number of steps in history
            prediction = max(forecast)  # as it is already a forecast, we get the maximum value for provisioning

            if prediction > (max(prev_usage) * self.LIMIT_FACTOR_PREV_USAGE):
                prediction = max(prev_usage) * self.LIMIT_FACTOR_PREV_USAGE  # to avoid strange behaviour at the beginning
            # if prediction < (min(prev_usage)*LIMIT_FACTOR_PREV_USAGE):
            #     print("strange min behaviour ", i * window*15)
            #     prediction = max(prev_usage) #to avoid strange behaviour at the beginning

        return prediction, forecast[:window_size]

    def recommend(self, trace, window_size=20, observation_window=(4 * 60) * 1):
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

    def dynamic_forecast_segment(self, trace, idx_init, idx_end, default_window, observation_window,prev_usage):
        observed_segment = min(idx_end, observation_window)  # choose the minimum number of steps between the maximum observation window and the available number of time steps
        observed_segment_index = range(idx_end - observed_segment, idx_end)
        step_test = observed_segment // 8  # the test size for testing cycles
        step_max = observed_segment // 3  # max cycle to be examined
        decision_thetascan = self.detect_behaviour(trace[observed_segment_index],default_window,step_test,step_max)

        if (decision_thetascan["scenario"][0] == "period" or decision_thetascan["scenario"][0] == "prob.period"):
            history = (decision_thetascan["period"].to_numpy())[0]  # Detected periodicity, get period as history for forecasting
        else:
            history = default_window  # get default 5 minute window as history for forecasting

        if (history > (idx_end// 2)):  # this only happens in the first 10 minutes maximum (warm-up)
            seen_trace = trace[(idx_end - history):idx_end]
            prediction = np.percentile(seen_trace,
                                       95)  # our prediction in the warm-up is the 95 percentile of the seen trace
            forecast = seen_trace[:default_window]  # our forecast in the warm-up is the seen trace
        else:
            forecast = self.next_step_forecasting_theta(trace[:idx_end],
                                                   history)  # based on the seen trace, predict the number of steps in history
            prediction = max(forecast)  # as it is already a forecast, we get the maximum value for provisioning

            if prediction > (max(prev_usage) * self.LIMIT_FACTOR_PREV_USAGE):
                prediction = max(prev_usage) * self.LIMIT_FACTOR_PREV_USAGE  # to avoid strange behaviour at the beginning
            # if prediction < (min(prev_usage)*LIMIT_FACTOR_PREV_USAGE):
            #     print("strange min behaviour ", i * window*15)
            #     prediction = max(prev_usage) #to avoid strange behaviour at the beginning

        return prediction, forecast, history

    def dynamic_recommend(self, trace, observation_window=(4 * 60) * 1):

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