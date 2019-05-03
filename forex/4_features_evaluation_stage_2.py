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

# currencyPair = 'USDJPY'
for currencyPair in currencyPairs:
    #%% Construct dataset for training
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    directory = path.join(dataDir, currencyPair)
    sys.stdout = open(path.join(directory, 'features_evaludation_stage_2_console_log.txt'), 'w')

    tmp = pd.read_csv(path.join(directory, 'features_evaluation_stage_1.csv'))
    tmp['Loss_Draw_Win'] = tmp['Loss_Draw_Win'].shift(-1)
    tmp.dropna(inplace=True)
    train, test = train_test_split(tmp, test_size=0.2, shuffle=True)
    print(len(tmp.index))
    print(len(train.index))
    print(len(test.index))

    featuresToBeUsed = tmp.columns.values[[colName.startswith('patterns_') for colName in tmp.columns.values]]
    # featuresToBeUsed = tmp.loc[:,patternCols].columns.values

    X_train = train[featuresToBeUsed].values
    Y_train = train['Loss_Draw_Win'].values
    train.to_csv(path.join(directory, 'features_evaluation_stage_2_train_set.csv'), sep=',', encoding='utf-8', float_format='%11.6f')

    X_test = test[featuresToBeUsed].values
    Y_test = test['Loss_Draw_Win'].values
    test.to_csv(path.join(directory, 'features_evaluation_stage_2_test_set.csv'), sep=',', encoding='utf-8', float_format='%11.6f')

    # del tmp

    #%% Random Forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib

    random_forest = RandomForestClassifier(n_estimators=100, n_jobs=16)
    print('Start training random forest')
    random_forest.fit(X_train, Y_train)
    print('Start evaluating the test set')
    Y_prediction = random_forest.predict(X_test)
    print('Scoring the accuracy')
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(round(acc_random_forest,2,), "%")
    joblib.dump(random_forest, path.join(directory, 'features_evaluation_stage_2_trained_random_forest.pkl'))


    #%% Feature Importance
    import numpy as np
    importances = pd.DataFrame({'feature':featuresToBeUsed,'importance':np.round(random_forest.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    # importances.iloc[:41,:].plot.bar()
    importances.to_csv(path.join(directory, 'features_evaluation_stage_2_random_forex_features_importances_matrix.csv'), sep=',', encoding='utf-8', float_format='%11.6f')


    #%%
    from sklearn.metrics import accuracy_score  #for accuracy_score
    from sklearn.model_selection import KFold #for K-fold cross validation
    from sklearn.model_selection import cross_val_score #score evaluation
    from sklearn.model_selection import cross_val_predict #prediction
    from sklearn.metrics import confusion_matrix #for confusion matrix

    print('--------------Cross Check The Accuracy of the model with KFold----------------------------')
    # print('The accuracy of the Random Forest Classifier is', round(accuracy_score(Y_prediction,Y_test)*100,2))
    kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
    result_rm=cross_val_score(random_forest, train[featuresToBeUsed], train['Loss_Draw_Win'], cv=10,scoring='accuracy')
    print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))

    #%%
    import matplotlib.pyplot as plt
    import seaborn as sns
    y_pred = cross_val_predict(random_forest, train[featuresToBeUsed], train['Loss_Draw_Win'], cv=10)
    sns.heatmap(confusion_matrix(train['Loss_Draw_Win'], y_pred),annot=True,fmt='3.0f',cmap="summer")
    plt.title('Confusion_matrix', y=1.05, size=15)
    plt.savefig(path.join(directory, 'features_evaluation_stage_2_random_forex_confusion_matrix.png'), bbox_inches='tight')
