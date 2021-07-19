#-------------------------------------------------------------------------------
# General Libraries
#-------------------------------------------------------------------------------
from matplotlib import pyplot as plt
#-------------------------------------------------------------------------------
# Other libraries and tools
#-------------------------------------------------------------------------------
from . import timeseries
from . timeseries import generate_ts

class ThetaScan():
    def __init__(self):
        self.ts_types = []

    def generate_ts_dataset(self, N, ts_len):
        self.ts_types = ["stationary", "trending", "periodic"]
        ts_dataset = []
        for i in range(N):
            ts_dataset.append(generate_ts(self.ts_types[i % len(self.ts_types)], ts_len))
        return ts_dataset