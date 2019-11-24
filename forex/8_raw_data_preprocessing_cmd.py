
#%% Initialize Global Settings
import platform
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import snappy
from fastparquet import ParquetFile
import os
import sys
import os.path as path
import random

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats

import talib

from forex.OHLC import OHLC
from forex.utils import *

dataDir = ''
if platform.system() == 'Windows':
    dataDir = 'F:\\modified_data'
elif platform.system() == 'Linux':
    dataDir = '/home/joseph/Desktop/datastore'
else:
    exit()

pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", '{:,.3f}'.format)

rollingWinSize = 600
batchSize = 32
epochNum = 1
weightFileName = 'raw_data_preprocessing_model.h5'
gpuId = 0
nnModleId = 2


#%% Settings Hyper Parameters and file paths
params = {
    'currency': sys.argv[1]
}

def hash(dictObj):
    import hashlib
    import json
    m = hashlib.md5()
    json_string = json.dumps(dictObj, sort_keys=True)
    m.update(json_string.encode('utf-8'))
    h = m.hexdigest()
    return h

raw_tick_data_csv_path = path.join(dataDir, params['currency']+'_2005-1-1_2019-5-18_ticks_new.csv')

tmp_preprocess_tick_df_path1 = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_1.parq')
tmp_preprocess_tick_df_path2 = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_2.parq')
tmp_preprocess_tick_df_path3 = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_3.parq')
tmp_preprocess_tick_df_path4 = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_4.parq')
tmp_preprocess_tick_df_path5 = path.join(dataDir, params['currency']+'_ticks_preprocess_tmp_5.parq')

def save_df(df, output_file):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
def read_df(input_file):
    pf = ParquetFile(input_file)
    df = pf.to_pandas()
    return df








#%% tick statistics
df = None
if not path.exists(tmp_preprocess_tick_df_path1):
    df = pd.read_csv(raw_tick_data_csv_path)
    print('Total Ticks: ' + str(len(df.index)))
    # df = df.iloc[0:100000]
    df.columns = ['Timestamp', 'Bid', 'Ask', 'BidVolume', 'AskVolume']
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
    df = df.round({'BidVolume': 3, 'AskVolume': 3})
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    save_df(df, tmp_preprocess_tick_df_path1)
else:
    df = read_df(tmp_preprocess_tick_df_path1)

#%% Check Range
if not path.exists(tmp_preprocess_tick_df_path2):
    # Financial econometric analysis at ultra-high frequency: Data
    # handling concerns
    # C.T. Brownlees∗, G.M. Gallo
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.1176&rep=rep1&type=pdf

    # A METHOD TO ‘‘CLEAN UP’’ ULTRA HIGH-FREQUENCY DATA*
    # Angelo M. Mineo**
    # Fiorella Romito**
    # file:///home/joseph/Downloads/999999_2007_0002_0036-151260.pdf

    # tmp2 = df['Bid'] - df['Bid'].shift(1)
    # bins = np.array([-0.006, -0.005, -0.004, -0.003, -0.002, -0.001, 0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    # bin_i = np.digitize(tmp2_val,bins,right=False)
    # unique, counts = np.unique(bin_i, return_counts=True)
    # np.asarray((unique, counts)).T
    # array([[        0,   5560038],
    #        [        1,   3323026],
    #        [        2,   8020891],
    #        [        3,   4912798],
    #        [        4,   8204253],
    #        [        5,  28597125],
    #        [        6,   7964122],
    #        [        7, 149363475],
    #        [        8,  27745821],
    #        [        9,   8218411],
    #        [       10,   5126276],
    #        [       11,   7962262],
    #        [       12,   3264986],
    #        [       13,   5706813]])

    count = 0
    k = 60
    phi_vol = 0.02
    phi = 0.00002
    if params['currency'] == 'USDJPY':
        phi = 0.002
    rollWin = k+1
    middleIndex = int(rollWin/2)
    def check_is_invalid_range_price(s):
        global count
        global phi
        if count % 1000000 == 0:
            print('count: ' + str(count/1000000))
        count += 1

        middleVal = s[middleIndex]
        seriesWithoutMiddleValue = np.delete(s, (middleIndex), axis=0)

        std = np.std(seriesWithoutMiddleValue)
        mean = np.mean(seriesWithoutMiddleValue)
        
        if abs(middleVal - mean) < 3 * std + phi:
            return False
        return True
    def check_is_invalid_range_volume(s):
        global count
        if count % 1000000 == 0:
            print('count: ' + str(count/1000000))
        count += 1

        middleVal = s[middleIndex]
        seriesWithoutMiddleValue = np.delete(s, (middleIndex), axis=0)

        std = np.std(seriesWithoutMiddleValue)
        mean = np.mean(seriesWithoutMiddleValue)
        
        if abs(middleVal - mean) < 3 * std + phi_vol:
            return False
        return True

    df['Is_Bid_Invalid_Range'] = df['Bid'].rolling(rollWin).apply(check_is_invalid_range_price, raw=True).shift(-1 * middleIndex)
    print('Bid Completed')
    count = 0

    df['Is_Ask_Invalid_Range'] = df['Ask'].rolling(rollWin).apply(check_is_invalid_range_price, raw=True).shift(-1 * middleIndex)
    print('Ask Completed')
    count = 0

    df['Is_BidVolume_Invalid_Range'] = df['BidVolume'].rolling(rollWin).apply(check_is_invalid_range_volume, raw=True).shift(-1 * middleIndex)
    print('Bid Volume Completed')
    count = 0

    df['Is_AskVolume_Invalid_Range'] = df['AskVolume'].rolling(rollWin).apply(check_is_invalid_range_volume, raw=True).shift(-1 * middleIndex)
    print('Ask Volume Completed')
    count = 0

    save_df(df, tmp_preprocess_tick_df_path2)
else:
    df = read_df(tmp_preprocess_tick_df_path2)


#%% Drop Invalid Data
if not path.exists(tmp_preprocess_tick_df_path3):
    total_ticks = len(df.index)
    print('Total Records: ' + str(total_ticks))
    invalid_df = df[(df['Is_Bid_Invalid_Range']==True) | (df['Is_Ask_Invalid_Range']==True) | (df['Is_BidVolume_Invalid_Range']==True) | (df['Is_AskVolume_Invalid_Range']==True)]
    total_invalid = len(invalid_df.index)
    print('Total Invalid: ' + str(total_invalid))
    print('Percentage: ' + str(total_invalid/total_ticks * 100))

    df.drop(index=invalid_df.index, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Time_Diff_In_Seconds'] = (df['Timestamp'] - df['Timestamp'].shift(1)).dt.total_seconds()

    save_df(df, tmp_preprocess_tick_df_path3)
else:
    df = read_df(tmp_preprocess_tick_df_path3)


#%% Data Validation
if not path.exists(tmp_preprocess_tick_df_path4):
    df['Is_Invalid_Timestamp'] = df['Time_Diff_In_Seconds'].map(lambda diff: (diff <= 0))
    df['Is_Bid_Less_Than_Or_Equal_To_0'] = df['Bid'].map(lambda x: (x <= 0))
    df['Is_BidVolume_Less_Than_Or_Equal_To_0'] = df['BidVolume'].map(lambda x: (x <= 0))

    save_df(df, tmp_preprocess_tick_df_path4)
else:
    df = read_df(tmp_preprocess_tick_df_path4)


#%% Check Gaps
if not path.exists(tmp_preprocess_tick_df_path5):
    rollWin = 10000
    df['Bid_Diff'] = (df['Bid'] - df['Bid'].shift(1)).apply(abs)
    df['Bid_Diff_Rolliing_Mean'] = df['Bid_Diff'].rolling(rollWin).apply(np.mean, raw=True).shift(1)
    df['Bid_Diff_Rolliing_Std'] = df['Bid_Diff'].rolling(rollWin).apply(np.std, raw=True).shift(1)
    df['Abs_Bid_Diff-Bid_Diff_Rolliing_Mean'] = (df['Bid_Diff'] - df['Bid_Diff_Rolliing_Mean']).apply(abs)
    df['Is_Data_Gap'] = (df['Time_Diff_In_Seconds'] > 900) & (df['Abs_Bid_Diff-Bid_Diff_Rolliing_Mean'] >= 2 * df['Bid_Diff_Rolliing_Std'])
    print('Total Data Gap: ' + str(len(df[df['Is_Data_Gap']==True].index)))

    # Different Methods to Clean Up Ultra High-Frequency Data(⋆)
    # http://www.old.sis-statistica.org/files/pdf/atti/rs08_spontanee_12_4.pdf
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html
    # http://www.librow.com/articles/article-7

    save_df(df, tmp_preprocess_tick_df_path5)
else:
    df = read_df(tmp_preprocess_tick_df_path5)

