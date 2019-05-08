#%% Initialize Global Settings
import platform
import pandas as pd
dataDir = ''
if platform.system() == 'Windows':
    dataDir = 'F:\\modified_data'
elif platform.system() == 'Linux':
    dataDir = '/home/joseph/Desktop/datastore'
else:
    exit()

pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", '{:,.3f}'.format)

import sys
import os.path as path
from forex.OHLC import OHLC

featureTypes = ['patterns']
currencyPairs = ['USDJPY', 'AUDUSD', 'EURGBP', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'XAUUSD']
if len(sys.argv) >= 2:
    featureTypes = [sys.argv[1]]
if len(sys.argv) >= 3:
    currencyPairs = [sys.argv[2]]
print(featureTypes)
print(currencyPairs)

if 'patterns' in featureTypes:
    for currencyPair in currencyPairs:
        directory = path.join(dataDir, currencyPair)
        # sys.stdout = open(path.join(directory, 'features_contruction_console_log.txt'), 'w')

        p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))
        p.set_df(p.get_df_with_resolution('1min'))
        p.df = p.df[['Timestamp', 'Close']]
        # p.df = p.df.iloc[:50000,:]
        # p.df.index = p.df['Timestamp']
        # print(p.df.head())

        #%% Introduce pattern recognition features from talib
        def detect_pattern_using_talib(prefix, o, h, l, c):
            results = {}

            import talib
            from inspect import getmembers, isfunction
            import numpy as np
            functs = [o for o in getmembers(talib) if isfunction(o[1])]
            for func in functs:
                funcName = func[0]
                if funcName.startswith('CDL'):
                    print('Computing Pattern Features Using talib: ' + prefix + '_' + funcName)
                    tmp = getattr(talib, funcName)(o, h, l, c) / 100
                    numberOfZero = np.count_nonzero(tmp==0)
                    if numberOfZero != len(tmp):
                        results[prefix+'_'+funcName] = tmp
            return pd.DataFrame(results)

        for resolStr in ['D', '6H', '3H', '1H', '30min', '15min', '10min', '5min', '1min']:
            tmp = p.df[['Close']].resample(resolStr, label='right').ohlc()
            patterns = detect_pattern_using_talib('patterns_'+resolStr, tmp['Close']['open'], tmp['Close']['high'], tmp['Close']['low'], tmp['Close']['close'])
            patterns['join_key'] = patterns.index
            p.df['join_key'] = p.df.index.floor(resolStr)
            p.df = pd.merge(p.df, patterns, on='join_key', how='left')
            p.df.index = p.df['Timestamp']
            p.df.drop('join_key', axis=1, inplace=True)

        p.df.drop('Close', axis=1, inplace=True)

        print('Remove all-zero/nan features')
        summary = p.df.describe()
        for colName in summary.columns.values:
            col = summary[[colName]]
            if ((col.loc['min',:] == col.loc['max',:]) & (col.loc['min',:] == 0)).bool():
                p.df.drop(colName, axis=1, inplace=True)
                print('Dropped: '+colName)

        print('drop rows without any signals, reduce csv size')
        print('Current number of rows: ' + str(len(p.df.index)))
        p.df.dropna(inplace=True)

        import numpy as np
        m = p.df.iloc[:,1:].values
        mask1 = [not np.all(m[i]==0) for i in range(0, len(p.df.index))]
        # mask2 = [np.all(np.isfinite(m[i])) for i in range(0, len(p.df.index))]
        # mask = mask1 and mask2
        p.df = p.df.loc[mask1,:]
        p.df.reset_index(drop=True, inplace=True)
        print('Drop Completed')
        print('Current number of rows: ' + str(len(p.df.index)))
        print('Saving pattern csv for '+currencyPair)

        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        p.save(path.join(directory, 'patterns.csv'))