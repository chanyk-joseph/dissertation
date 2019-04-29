
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

#%% Read data
import os.path as path
import pandas as pd
from forex.OHLC import OHLC

currencyPair = 'USDJPY'
p = OHLC(path.join(dataDir, currencyPair+'_1MIN_(1-1-2008_31-12-2017).csv'))
p.df.drop(['Timestamp'], axis=1)

#%% Convert to 1min interval and fill missing records with previous values
p.set_df(p.get_df_with_resolution('1min'))

#%% Calculate pip returns within future X mins
p.merge_df(p.get_mins_returns_cols([1,2,4,8,16,32], 'mins'))
p.df.dtypes

#%% 
p.df.drop(['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)

#%% 
pd.DataFrame({
    '1mins': p.df['PIP_Return_forward_looking_for_1mins_max'],
    '2mins': p.df['PIP_Return_forward_looking_for_2mins_max'],
    '4mins': p.df['PIP_Return_forward_looking_for_4mins_max'],
    '8mins': p.df['PIP_Return_forward_looking_for_8mins_max'],
    '16mins': p.df['PIP_Return_forward_looking_for_16mins_max'],
    '32mins': p.df['PIP_Return_forward_looking_for_32mins_max']
})

#%% Maximum actual returns within following mins in the future
import numpy as np
key = 'max'
arrKeys = ['PIP_Return_forward_looking_for_1mins_'+key,
    'PIP_Return_forward_looking_for_2mins_'+key,
    'PIP_Return_forward_looking_for_4mins_'+key,
    'PIP_Return_forward_looking_for_8mins_'+key,
    'PIP_Return_forward_looking_for_16mins_'+key,
    'PIP_Return_forward_looking_for_32mins_'+key
]

indexes = []
results = {'mean':[], 'std':[], '75%':[], '95%':[]}
summary = p.df.loc[:, arrKeys].describe()
for resol in [1,2,4,8,16,32]:
    indexes.append(str(resol)+'mins')
    resolStr = 'PIP_Return_forward_looking_for_' + str(resol) + 'mins'
    for stat in ['mean', 'std', '75%']:
        results[stat].append(summary[resolStr+'_'+key][stat])
    results['95%'].append(np.percentile(p.df[resolStr+'_'+key].dropna(), 95))
newdf = pd.DataFrame(results, index=indexes)
newdf

#%% Get the percentile boundary
import numpy as np
percentiles = ['3', '16', '50', '84', '97']
results = {}
colToBeUsed = p.df['PIP_Return_forward_looking_for_32mins_max'].dropna()
for stat in percentiles:
    results[stat] = np.percentile(colToBeUsed, int(stat))
results


#%% Convert percentile into corresponding signal class: [
# 1 (0%  <= x <  3%), 
# 2 (3%  <= x <  16%), 
# 3 (16% <= x <  50%), 
# 4 (50% <= x <  84%), 
# 5 (84% <= x <  97%), 
# 6 (97% <= x <= 100%)
# ]

from IPython.core.display import Image, display
display(Image(url="https://timotheories.files.wordpress.com/2017/01/standard_deviation_diagram-svg.png?w=1000&h=700&crop=1", width=800, height=500))

import pandas as pd
bins = [-9999999]
for percentile in percentiles:
    bins.append(results[percentile])
bins.append(9999999)
# bins
p.df['Signal'] = pd.cut(p.df['PIP_Return_forward_looking_for_32mins_max'], bins, labels=[1, 2, 3, 4, 5, 6])
p.df[['Signal']]

#%% Check the distribution of [1, 6] signal classes
signalDf = p.df.groupby(p.df['Signal']).size().to_frame()
signalDf.columns = ['Count']
signalDf.plot()
signalDf


#%% Get backtest results of all signals
allTrades = []
df = p.df[['Timestamp', 'Close', 'Signal']].copy()
dfDict = df.to_dict('split')
for i in range(1, 7):
    print('============= Signal_'+str(i)+' Performance ==============')
    tradeDf = p.backtest(i, dfDict)
    print('Final Balance: ' + str(float(tradeDf[-1:].Balance)))
    tradeDf[['Balance']].plot()
    allTrades.append(tradeDf)

#%% Findout which signal to be used, i.e. maximize profit while minimize drawdown
results = []
indexes = []
for i in range(0, 6):
    trade = allTrades[i]
    results.append({
        'Final Balance': float(trade[-1:].Balance),
        'Max Drawdown': float(trade.describe()[['Drawdown']].loc['min'])
    })
    indexes.append('Signal_' + str(i+1))
tmpDf = pd.DataFrame(results, index=indexes)
tmpDf[['Final Balance', 'Max Drawdown']].plot()
tmpDf

#%% 
# mergedDf.plot()
allTrades[5].to_csv(path.join(dataDir, 'signal6.csv'), sep=',', encoding='utf-8', index=False)

#%% talib
import talib
df = p.df.loc[:, ['Open', 'High', 'Low','Close']].copy().dropna()
df['CDLTRISTAR'] = talib.CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Close'])
df[['CDLTRISTAR']].describe()
df

#%% Get all pattern detection algorithm from talib
from inspect import getmembers, isfunction
functs = [o for o in getmembers(talib) if isfunction(o[1])]
for func in functs:
    funcName = func[0]
    if funcName.startswith('CDL'):
        print('Computing Pattern Features Using talib: ' + funcName)
        df[funcName] = getattr(talib, funcName)(df['Open'], df['High'], df['Low'], df['Close']) / 100
tmp = df.describe().T
tmp

#%% 
df.info()

