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

