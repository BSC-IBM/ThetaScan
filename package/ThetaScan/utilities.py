import numpy as np
from scipy import signal
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm

from ThetaScan import utilities_cy

def generate_ts(ts_type, ts_len, usage_mean=200, usage_std=10, spike_mean=800, spike_ratio=0.5):
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
    ts_dataset = []
    ts_class = []
    for i in range(N):
        ts_dataset.append(generate_ts(ts_types[i%len(ts_types)], ts_len))
        ts_class.append(ts_types[i%len(ts_types)])
    return ts_dataset, ts_class

def plot_trace(trace, plt_name="trace", y_label="CPU (milicores)", trace_legend="CPU Usage"):
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
    nominator = np.abs(y_test - y_pred)
    denominator = np.abs(y_test) + np.abs(y_pred)
    return np.mean(2.0 * nominator / denominator) #the 2 in the nominator is because of symmetry

@profile
def convolution_filter(y, period):
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

@profile
def compute_ses(y, alpha):
    nobs = len(y)  # X from the slides

    # Forecast Array
    fh = np.full(nobs + 1, np.nan)  # Initialize the Forecast array to NaNs # S from the slides
    fh[0] = y[0]  # Initialization of first value (instead of NaN)
    fh[1] = y[0]  # Initialization of first forecast

    # Simple Exponential Smoothing
    for t in range(2, nobs + 1):
        fh[t] = alpha * y[t - 1] + (1 - alpha) * fh[t - 1]  # s[t] = alpha * y....

    return (fh[:nobs], fh[nobs])

@profile
def forecast_ses(fh_next, start, end):
    ## Forecast Array
    fh_forecast = np.full(end - start, np.nan)
    fh_forecast[:] = fh_next

    return fh_forecast

@profile
def seasonal_decompose(y, period):
    nobs = len(y)

    # At least two observable periods in the trace
    if nobs < 2 * period:
        raise ValueError('lengh of signal must be larger than (2 * period)')

    # Convolution to retrieve step-by-step trend
    trend = convolution_filter(y, period)

    # Multiplicative de-trending to Retrieve average Season (period pattern)
    detrended = y / trend
    #period_averages = np.array([np.nanmean(detrended[i::period], axis=0) for i in range(period)])
    period_averages = utilities_cy.compute_period_averages(detrended, period)

    period_averages /= np.mean(period_averages, axis=0)

    return period_averages  # "season" for deseasonalize

@profile
def deseasonalize(y, season):
    nobs = len(y)
    period = len(season)

    seasonal = np.tile(season, (nobs // period) + 1).T[:nobs]
    return y / seasonal

@profile
def reseasonalize(y, season, start):
    nobs = len(y)
    period = len(season)

    shift = period - (start % period)
    season = np.concatenate((season[-shift:], season[:-shift]))

    seasonal = np.tile(season, (nobs // period) + 1).T[:nobs]
    return y * seasonal

@profile
def compute_trend(y):
    lm = np.polyfit(np.arange(len(y)), y, 1)

    slope = lm[0]
    intercept = lm[1]
    drift = (slope * np.arange(0, len(y))) + intercept

    return (slope, intercept, drift)

@profile
def retrend(y, start, end, slope, intercept):
    drift = (slope * np.arange(start, end)) + intercept

    pred = y * (drift / np.mean(y))  # CHECK - Find a better way to estimate the general reconstruction...
    return pred
