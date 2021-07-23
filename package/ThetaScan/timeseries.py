import numpy as np

def generate_ts(ts_type, ts_len, usage_mean=200, usage_std=100, slope=0.1, period=20, spike_mean=800, spike_ratio=0.5):
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

