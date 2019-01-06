from strategies.strategy import Strategy
import talib
import numpy as np

class MACD(Strategy):
    def __init__(self):
        super().__init__("MACD")

        self.params = {
            "fastperiod": 12,
            "slowperiod": 26,
            "signalperiod": 9
        }

    def generate_signal(self, df):
        macd, macdsignal, macdhist = talib.MACD(df['Close'], self.params["fastperiod"], self.params["slowperiod"], self.params["signalperiod"])
        
        signal = np.sign(np.sign(macdhist).diff()) # Sell: -1; Hold: 0; Buy: 1
        hist_slope = macdhist.diff()
        hist_slope_min = hist_slope.min()
        hist_slope_max = hist_slope.max()
        def calculate_strength(slope):
            if slope == np.nan:
                return np.nan
            if slope >= 0:
                return slope / hist_slope_max
            else:
                return abs(slope / hist_slope_min)
        signal_strength = hist_slope.apply(calculate_strength)
        return signal, signal_strength

    def generate_buy_sell_records(self, df):
        signal, signal_strength = self.generate_signal(df)

        def to_buy_sell(val):
            if val > 0:
                return 1
            elif val < 0:
                return -1
            return 0
        df["MACD"] = (signal * signal_strength).apply(to_buy_sell)
        return df