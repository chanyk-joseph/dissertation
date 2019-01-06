import multiprocessing
import glob
import pandas as pd

from os import path
from forex.utils import parseDateTime

def normalize_ohlc_csv(csvFile):
    df = pd.read_csv(csvFile)
    df['Timestamp'] = df['Timestamp'].apply(lambda x: parseDateTime(x).isoformat().replace("+00:00", "Z"))
    return df

def update(csvFile):
    outputFolder = path.join("F:\\modified_data")
    filename = path.basename(csvFile)
    outputFilePath = path.join(outputFolder, filename)

    df = normalize_ohlc_csv(csvFile)
    df.to_csv(outputFilePath, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=6)

    csvFiles = glob.glob("F:\\raw_data\\*.csv")
    for csvFile in csvFiles:
        pool.apply_async(update, (csvFile, ))

    pool.close()
    pool.join()