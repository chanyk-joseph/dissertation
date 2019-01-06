from strategies.strategy import Strategy
import datetime
import talib
import numpy as np

class SMA(Strategy):
    def __init__(self, duration_in_seconds):
        super().__init__()
        self.duration_in_seconds = duration_in_seconds

    def calculate(self, df, end_datetime):
        from_datetime = end_datetime - datetime.timedelta(seconds=self.duration_in_seconds)
        sub_df = self.get_dataframe_by_time_arange(df, from_datetime, end_datetime)
        if len(sub_df.index) == 0:
            return np.nan
        result = talib.MA(sub_df["Close"], len(sub_df.index), matype=0)
        if len(result) == 0 or result.iloc[-1] == np.nan:
            return np.nan
        return result.iloc[-1]