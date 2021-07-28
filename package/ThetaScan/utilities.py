import numpy as np
import pandas as pd
from scipy import signal
import statsmodels
import statsmodels.api as sm
import sktime
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.forecasting.trend import PolynomialTrendForecaster
from warnings import filterwarnings
try:
    from sktime.transformations.series.detrend import Detrender
except:
    from sktime.transformers.single_series.detrend import Detrender

def generate_ts(ts_type, ts_len, usage_mean=200, usage_std=10, slope=0.15, period=20, spike_mean=800, spike_ratio=0.5):
    trace = np.zeros(ts_len)

    if ts_type == "stationary":
        trace = np.random.normal(usage_mean, usage_std, ts_len)
        trace = np.clip(trace, 0.1,None)
    elif ts_type == "trending":
        trend_ts = usage_mean * slope * np.arange(ts_len) / period
        trace = np.random.normal(usage_mean, usage_std, ts_len)
        trace = trace + trend_ts
    elif ts_type == "periodic":
        periodic_ts = np.random.normal(usage_mean, usage_std, ts_len)
        trace_idx = np.arange(ts_len)
        periodic_cycle = period / spike_ratio
        trace_idx = trace_idx % periodic_cycle
        spike_mask = np.zeros(ts_len)
        spike_mask[trace_idx < period] = 1
        trace = periodic_ts + spike_mask * spike_mean
        trace = np.clip(trace, 0.1, None) #added clipping to avoid zero and negative values
    else:
        print("The specified time series behavior type {} is not recognized. Return all zero time series!".format(ts_type))

    return trace

def generate_ts_dataset(N, ts_len):
    ts_types = ["stationary", "trending", "periodic"]
    ts_dataset = []
    for i in range(N):
        ts_dataset.append(generate_ts(ts_types[i%len(ts_types)], ts_len))
    return ts_dataset

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


def forecast_ses(fh_next, start, end):
    ## Forecast Array
    fh_forecast = np.full(end - start, np.nan)
    fh_forecast[:] = fh_next

    return fh_forecast


def seasonal_decompose(y, period):
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
    nobs = len(y)
    period = len(season)

    seasonal = np.tile(season, (nobs // period) + 1).T[:nobs]
    return y / seasonal


def reseasonalize(y, season, start):
    nobs = len(y)
    period = len(season)

    shift = period - (start % period)
    season = np.concatenate((season[-shift:], season[:-shift]))

    seasonal = np.tile(season, (nobs // period) + 1).T[:nobs]
    return y * seasonal


def compute_trend(y):
    lm = np.polyfit(np.arange(len(y)), y, 1)

    slope = lm[0]
    intercept = lm[1]
    drift = (slope * np.arange(0, len(y))) + intercept

    return (slope, intercept, drift)


def retrend(y, start, end, slope, intercept):
    drift = (slope * np.arange(start, end)) + intercept

    pred = y * (drift / np.mean(y))  # CHECK - Find a better way to estimate the general reconstruction...
    return pred
