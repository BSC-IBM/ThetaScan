import numpy as np
from scipy import signal
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm


def generate_ts(ts_type, ts_len, usage_mean=200, usage_std=10, spike_mean=800, spike_ratio=0.5):
    ''' Generate a synthetic trace.
        Parameters
        ----------
        ts_type (string): type of the synthetic trace that we want to generate (stationary, trending, periodic or sinusoidal).
        usage_mean (float): mean resource usage
        usage_std (float): standard deviation of the resource usage
        spike_ratio (float): ratio at which resource spikes appear
    '''
    trace = np.zeros(ts_len)

    if ts_type == "stationary":
        usage_mean = random.randint(100,1000)
        print("Generating stationary trace centered at {}...".format(usage_mean))
        trace = np.random.normal(usage_mean, usage_std, ts_len)
        trace = np.clip(trace, 0.1,None)
    elif ts_type == "trending":
        usage_mean = random.randint(100, 1000)
        slope = random.randint(10,90)/ 100
        print("Generating trending trace with slope {} and intercept {}...".format(slope, usage_mean))
        trend_ts = usage_mean + slope * np.arange(ts_len)
        trace = np.random.normal(usage_mean, usage_std, ts_len)
        trace = trace + trend_ts
    elif ts_type == "periodic":
        period = random.randint(20,40)
        print("Generating periodic trace with period {}".format(period))
        periodic_ts = np.random.normal(usage_mean, usage_std, ts_len)
        trace_idx = np.arange(ts_len)
        periodic_cycle = period / spike_ratio
        trace_idx = trace_idx % periodic_cycle
        spike_mask = np.zeros(ts_len)
        spike_mask[trace_idx < period] = 1
        trace = periodic_ts + spike_mask * spike_mean
        trace = np.clip(trace, 0.1, None) #added clipping to avoid zero and negative values
    elif ts_type == "sinusoidal":
        trace_idx = np.arange(ts_len)
        noise = np.random.normal(usage_mean, usage_std, ts_len)
        freq = 1 / random.randint(20,40)
        trace = usage_mean*np.sin(2*np.pi*freq*trace_idx) + noise
    else:
        print("The specified time series behavior type {} is not recognized. Return all zero time series!".format(ts_type))

    return trace

def generate_ts_dataset(N, ts_len, ts_types = ["stationary", "trending", "periodic"] ):
    ''' Generate a synthetic dataset.
        Parameters
        ----------
        N (int): number of traces in the dataset
        ts_len (int): length of the time series
        ts_types (list): types of the synthetic trace that we want to generate (stationary, trending, periodic or sinusoidal).
    '''
    ts_dataset = []
    ts_class = []
    for i in range(N):
        ts_dataset.append(generate_ts(ts_types[i%len(ts_types)], ts_len))
        ts_class.append(ts_types[i%len(ts_types)])
    return ts_dataset, ts_class

def plot_trace(trace, plt_name="trace", y_label="CPU (milicores)", trace_legend="CPU Usage"):
    ''' 
    Plot a resource usage trace
    Parameters
    ----------
    trace (numpy array): resource usage trace that we want to plot
    plt_name (string): name of the plot
    y_label (string): label of the y axis
    trace_legend (string): legend of the plot
    '''
    trace_len = len(trace)
    trace_idx = np.arange(trace_len) * 15 #set index 0,15,30,45,...,7470,7485
    fig, ax = plt.subplots()
    #trace_pd = pd.DataFrame({trace_legend: trace}, index=trace_idx)
    plt.plot(trace_idx,trace)
    plt.xlabel("Time (seconds)")
    plt.ylabel(y_label)
    plt.title(plt_name)
    #if plt_name == "trending trace": ax.legend(loc='lower right')
    #else: ax.legend(loc='upper right')
    plt.show()

def plot_recommendations(trace, forecast, request, plt_name="trace", y_label="CPU (milicores)", trace_legend="CPU Usage"):
    ''' 
    Plot recommendations for the resource usage trace
    
    Parameters
    ----------
    trace (numpy array): resource usage trace that we want to plot
    forecast (numpy array): granular forecast of the trace
    request (numpy array): requested resource trace 
    plt_name (string): name of the plot
    y_label (string): label of the y axis
    trace_legend (string): legend of the plot
    
    '''
    
    trace_len = len(trace)
    trace_idx = np.arange(trace_len) * 15 #set index 0,15,30,45,...,7470,7485

    fig, ax = plt.subplots()
    plt.plot(trace_idx, trace, label = trace_legend)
    plt.plot(trace_idx, request, label = "ThetaScan request")
    plt.plot(trace_idx, forecast, label = "ThetaScan forecast")
    #trace_pd = pd.DataFrame({trace_legend: trace,"ThetaScan forecasted request": request, "ThetaScan forecasted predicted": forecast}, index=trace_idx)
    plt.xlabel("Time (seconds)")
    plt.ylabel(y_label)
    plt.title(plt_name)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #fig = plt.gcf()
    fig.set_size_inches(10, 4)
    plt.show()

def smape_loss(y_pred,y_test):
    ''' 
    Symmetric mean absolute percentage error
    
    Parameters
    ----------
    y_pred (numpy array): prediction 
    y_test (numpy array): real time series
    
    '''
    nominator = np.abs(y_test - y_pred)
    denominator = np.abs(y_test) + np.abs(y_pred)
    return np.mean(2.0 * nominator / denominator) #the 2 in the nominator is because of symmetry


def convolution_filter(y, period):
    ''' 
    Convolution filter
    
    Parameters
    ----------
    y (numpy array): time series 
    period (int): period in the time series
    
    Returns
    -------
    conv_signal (numpy array): convolved signal
    
    '''
    # Prepare Filter
    if period % 2 == 0:
        filt = np.array([.5] + [1] * (period - 1) + [.5]) / period
    else:
        filt = np.repeat(1. / period, period)

    # Signal Convolution
    conv_signal = signal.convolve(y, filt, mode='valid')

    # Padding (2-Sided Convolution)
    trim_head = int(np.ceil(len(filt) / 2.) - 1) or None
    trim_tail = int(np.ceil(len(filt) / 2.) - len(filt) % 2) or None

    if trim_head:
        conv_signal = np.r_[conv_signal, [np.nan] * trim_tail]
    if trim_tail:
        conv_signal = np.r_[[np.nan] * trim_head, conv_signal]

    return conv_signal


def compute_ses(y, alpha):
    ''' 
    Compute the Simple Exponential Smoothing (SES)
    
    Parameters
    ----------
    y (numpy array): time series 
    alpha (float): constant for computing SES
    
    Returns
    -------
    fh[:nobs] (numpy array): filtered signal
    
    '''
    nobs = len(y)  # X from the slides

    # Forecast Array
    fh = np.full(nobs + 1, np.nan)  # Initialize the Forecast array to NaNs # S from the slides
    fh[0] = y[0]  # Initialization of first value (instead of NaN)
    fh[1] = y[0]  # Initialization of first forecast

    # Simple Exponential Smoothing
    for t in range(2, nobs + 1):
        fh[t] = alpha * y[t - 1] + (1 - alpha) * fh[t - 1]  # s[t] = alpha * y....

    return (fh[:nobs], fh[nobs])


def forecast_ses(fh_next, start, end):
    ''' 
    Forecast using the SES model
    
    Parameters
    ----------
    fh_next (numpy array): forecast of the next forecasting horizon
    
    Returns
    -------
    fh_forecast (numpy array): forecast
    
    '''
    ## Forecast Array
    fh_forecast = np.full(end - start, np.nan)
    fh_forecast[:] = fh_next

    return fh_forecast


def seasonal_decompose(y, period):
    ''' 
    Decompose the time series into the seasonal component 
    
    Parameters
    ----------
    y (numpy array): time series
    period (int): period of the time series
    
    Returns
    -------
    period_averages (numpy array): period pattern
    
    '''
    nobs = len(y)

    # At least two observable periods in the trace
    if nobs < 2 * period:
        raise ValueError('lengh of signal must be larger than (2 * period)')

    # Convolution to retrieve step-by-step trend
    trend = convolution_filter(y, period)

    # Multiplicative de-trending to Retrieve average Season (period pattern)
    detrended = y / trend
    period_averages = np.array([np.nanmean(detrended[i::period], axis=0) for i in range(period)])
    period_averages /= np.mean(period_averages, axis=0)

    return period_averages  # "season" for deseasonalize


def deseasonalize(y, season):
    ''' 
    Remove the seasonal component of a time series
    
    Parameters
    ----------
    y (numpy array): time series
    season (numpy array): period pattern
    
    Returns
    -------
    (numpy array): time series without the seasonal component
    
    '''
    nobs = len(y)
    period = len(season)

    seasonal = np.tile(season, (nobs // period) + 1).T[:nobs]
    return y / seasonal


def reseasonalize(y, season, start):
    ''' 
    Add a seasonal component to a time series 
    
    Parameters
    ----------
    y (numpy array): time series
    season (numpy array): period pattern
    start (int): index where to start adding the season
    
    Returns
    -------
    (numpy array): time series with the seasonal component
    '''
    nobs = len(y)
    period = len(season)

    shift = period - (start % period)
    season = np.concatenate((season[-shift:], season[:-shift]))

    seasonal = np.tile(season, (nobs // period) + 1).T[:nobs]
    return y * seasonal


def compute_trend(y):
    ''' 
    Compute the trend of a time seris
    
    Parameters
    ----------
    y (numpy array): time series
    
    Returns
    -------
    slope (int): slope of the time series
    intercept (int): slope of the time series 
    drift (int): drift of the time series
    
    '''
    lm = np.polyfit(np.arange(len(y)), y, 1)

    slope = lm[0]
    intercept = lm[1]
    drift = (slope * np.arange(0, len(y))) + intercept

    return (slope, intercept, drift)


def retrend(y, start, end, slope, intercept):
    ''' 
    Add the trending component of the time series
    
    Parameters
    ----------
    y (numpy array): time series
    start (int): starting index of the retrending
    end (int): ending index of the retrending
    intercept (int): intercept of the trend
    
    Returns
    -------
    pred (numpy array): time series with the trend added
    '''
    drift = (slope * np.arange(start, end)) + intercept

    pred = y * (drift / np.mean(y))  
    return pred
