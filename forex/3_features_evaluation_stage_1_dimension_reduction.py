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


currencyPairs = ['AUDUSD', 'EURGBP', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'XAUUSD']
if len(sys.argv) > 1:
    currencyPairs = [sys.argv[1]]
print(currencyPairs)

for currencyPair in currencyPairs:
    #%% Read both signal and pattern data
    directory = path.join(dataDir, currencyPair)

    signalsDf = pd.read_csv(path.join(directory, 'signals.csv'))
    patternssDf = pd.read_csv(path.join(directory, 'patterns.csv'))
    mergedDf = pd.merge(signalsDf, patternssDf, on='Timestamp', how='left')

    #%% Remove all-zero/nan features
    summary = mergedDf.describe()
    for colName in summary.columns.values:
        col = summary[[colName]]
        if ((col.loc['min',:] == col.loc['max',:]) & (col.loc['min',:] == 0)).bool():
            mergedDf.drop(colName, axis=1, inplace=True)
            print('Dropped: '+colName)
    mergedDf

    #%% Generate correlation matrix of all columns
    corrMatrix = mergedDf.corr()
    corrMatrix.to_csv(path.join(directory, 'features_evaluation_stage_1_correlation_matrix.csv'), sep=',', encoding='utf-8', float_format='%11.6f')

    #%% Generate heat map to visualize correlation matrix of subset of features
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.clf()
    sns.heatmap(mergedDf.iloc[:,11:41].corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
    fig=plt.gcf()
    fig.set_size_inches(20,12)
    plt.savefig(path.join(directory, 'features_evaluation_stage_1_correlation_col_11_to_40_before.png'), bbox_inches='tight')


    #%% Remove features that has 1 correlation with previous features, i.e. remove duplication
    removedFeatureNames = []
    allCols = corrMatrix.columns.values
    for i in range(0, len(allCols)-1):
        colName_i = allCols[i]
        if colName_i in removedFeatureNames:
            continue
        for j in range(i+1, len(allCols)):
            colName_j = allCols[j]
            corr_val = corrMatrix[colName_i][colName_j]
            if corr_val == 1:
                removedFeatureNames.append(colName_j)
    print('Dropped features with correlation 1:')
    print(removedFeatureNames)
    mergedDf.drop(removedFeatureNames, axis=1, inplace=True)

    #%% Generate the correlation heat map again to see if there are still duplications
    plt.clf()
    sns.heatmap(mergedDf.iloc[:,11:41].corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
    fig=plt.gcf()
    fig.set_size_inches(20,12)
    plt.savefig(path.join(directory, 'features_evaluation_stage_1_correlation_col_11_to_40_after.png'), bbox_inches='tight')

    mergedDf.round(6)
    mergedDf.to_csv(path.join(directory, 'features_evaluation_stage_1.csv'), sep=',', encoding='utf-8', index=False, float_format='%11.6f')
