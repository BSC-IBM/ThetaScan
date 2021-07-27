def thetascan_forecast(trace, window, i, observation_window, prev_usage):
    '''
    :param trace: trace to examine
    :param window: window size to provision (5 minute period inside vpa_recommender, 20 timesteps)
    :param i: window index to provision
    :param observation_window: observation window to assess periodicity
    :param prev_usage: usage in previous time window
    '''
    # Detect stationarity, periodicity, trending
    observed_segment = min(i * window,
                           observation_window)  # choose the minimum number of steps between the maximum observation window and the available number of time steps
    observed_segment_index = range(i * window - observed_segment, i * window)
    step_test = observed_segment // 8  # the test size for testing cycles
    step_max = observed_segment // 3  # max cycle to be examined
    decision_thetascan = make_decision_limit(trace[observed_segment_index], limit=window, test_size=step_test,
                                             sp_max=step_max)

    LIMIT_FACTOR_PREV_USAGE = 3  # we can change this constant, we use it to avoid forecasting peaks

    if (decision_thetascan["scenario"][0] == "period" or decision_thetascan["scenario"][0] == "prob.period"):
        history = (decision_thetascan["period"].to_numpy())[
            0]  # Detected periodicity, get period as history for forecasting
    else:
        history = window  # get default 5 minute window as history for forecasting

    if (history > (i * window // 2)):  # this only happens in the first 10 minutes maximum (warm-up)
        seen_trace = trace[(i * window - history):i * window]
        prediction = np.percentile(seen_trace,
                                   95)  # our prediction in the warm-up is the 95 percentile of the seen trace
        forecast = seen_trace[:window]  # our forecast in the warm-up is the seen trace
    else:
        forecast = next_step_forecasting_theta(trace[:i * window],
                                               history)  # based on the seen trace, predict the number of steps in history
        prediction = max(forecast)  # as it is already a forecast, we get the maximum value for provisioning

        if prediction > (max(prev_usage) * LIMIT_FACTOR_PREV_USAGE):
            prediction = max(prev_usage) * LIMIT_FACTOR_PREV_USAGE  # to avoid strange behaviour at the beginning
        # if prediction < (min(prev_usage)*LIMIT_FACTOR_PREV_USAGE):
        #     print("strange min behaviour ", i * window*15)
        #     prediction = max(prev_usage) #to avoid strange behaviour at the beginning

    return prediction, forecast[:window]

def make_decision_limit(trace, limit, se_threshold = 0.01, test_size = 1000, sp_max = 500):
    scenarios = ['stat', 'prob.ctan', 'prob.period', 'period']

    ## 1. Stationarity Test
    stationary_1 = StationarityTest(trace)

    ## 2. Our Test
    period_est, error_est = EstimateCycles(trace, test_size = test_size, sp_max = sp_max)
    stationary_2 = np.var(error_est) < se_threshold

    ## 3. Produce recommendation
    if stationary_2 :
        if stationary_1 : scenario = 0      ## ADF = Ours = STAT -> Stationary
        else : scenario = 1                 ## ADF = NonStat / Ours = STAT -> Most of them CTAN (full or partial)
    else :
        if stationary_1 : scenario = 2      ## ADF = Stat / Ours = NonStat -> Time-Dependant or Periodic with low period (ADF false positiva)
        else : scenario = 3                 ## ADF = NonStat / Ours = NonStat -> Proceed to Predict, Periodic
    y_notrend, y_pred, slope = TrendingDetection(trace)

    if period_est < limit:
        scenario = 0

    ## 4. Return all flags/estimations
    information = (stationary_1, stationary_2, True, scenarios[scenario], np.mean(trace), np.std(trace), np.max(trace), period_est, slope)

    return pd.DataFrame([information], columns=['adf.stat', 'our.stat', 'parsed', 'scenario', 'mean', 'st.dev', 'max', 'period', 'slope'])

def StationarityTest (trace, threshold = 0.01):
    adf = sm.tsa.stattools.adfuller(trace)
    stationary = adf[1] < threshold
    return stationary

'''
trace     : the trace to examine
test_size : the test size for testing cycles
sp_max    : max cycle to be examined
returns
- period       : the estimated periodicity
- period_error : the errors for each periodicity test
'''
def EstimateCycles(trace, test_size=1000, sp_max=1000, verbose=False):
    filterwarnings('ignore')

    y_min = 0
    y_max = len(trace)

    if test_size >= len(trace):
        test_size = len(trace) // 2

    y_train, y_test = temporal_train_test_split(trace[y_min:y_max], test_size=test_size)

    errors, sp_values = ciclicity_errors_given_pair(y_train, y_test, sp_max=sp_max)
    # period = result.index(min(result))
    period = sp_values[np.argmin(errors)]
    period_error = pd.Series(errors)
    period_error.name = "Period error"
    return (period, period_error)

def ciclicity_errors_given_pair(y_train,
                                y_test,
                                f_loss=smape_loss,
                                forecaster=ThetaModel,
                                sp_max=30):
    errors = []
    sp_values = list(range(1, min(sp_max, len(y_train) // 2)))

    if not isinstance(y_train, pd.DataFrame):
        y_train = pd.Series(y_train)

    if not isinstance(y_test, pd.DataFrame):
        y_test = pd.Series(y_test, index=range(len(y_train),len(y_train)+len(y_test)))

    for sp_val in sp_values:
        fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
        forecaster_ = forecaster()  # monthly seasonal periodicity
        forecaster_.fit(y_train, sp = sp_val, alpha = 0.2)
        y_pred = forecaster_.forecast(fh)[1]
        y_pred = pd.Series(y_pred, index = y_test.index)
        current_error = f_loss(y_pred, y_test)
        errors.append(current_error)

    return errors, sp_values

def TrendingDetection(trace, degree=1):
    if not isinstance(trace, pd.DataFrame):
        trace = pd.Series(trace)

    forecaster = PolynomialTrendForecaster(degree=degree)
    transformer = Detrender(forecaster=forecaster)

    fh_ins = -np.arange(len(trace))  # in-sample forecasting horizon
    trace_notrend = transformer.fit_transform(trace)
    trace_pred = forecaster.fit(trace).predict(fh=fh_ins)

    trace_notrend = pd.Series(trace_notrend, name="No Trend")
    trace_pred = pd.Series(trace_pred, name="Fitted Trend")

    slope = (trace_pred[trace_pred.index[-1]] - trace_pred[trace_pred.index[0]]) / len(trace_pred)

    return (trace_notrend, trace_pred, slope)

def next_step_forecasting_theta (trace, step):
    if not isinstance(trace, pd.DataFrame):
        trace = pd.Series(trace)

    if step < 1:
        y_pred = [0]
    else:
        forecaster_4 = ThetaModel()
        forecaster_4.fit(trace, step, alpha=0.2)
        fh = np.arange(step) + 1  ## <- LENGTH WILL BE HIST (fit) + NEXT STEP (predict)
        y_pred = forecaster_4.forecast(fh)[1]  ## <- PREDICT STEP

    return y_pred
