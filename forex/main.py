import os.path as path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import talib

from forex.utils import *

def plot(df):
    df.plot(x='datetime', y='total_asset_value')
    plt.show()

if __name__ == "__main__":
    print(parseDateTimeStr('20080103 23:01:01:100'))

# script_dir = path.dirname(path.realpath(sys.argv[0]))
# csvFile = path.join(script_dir, 'trade-record-simple.csv')

# df = pd.read_csv(csvFile)
# plot(df)

# import talib
# close = np.random.random(100)
# SMA = talib.MA(close,30,matype=0)

# plt.plot(SMA)
# plt.ylabel('some numbers')
# plt.show()