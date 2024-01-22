import numpy as np
import pandas as pd

class TechnicalAnalysis:

    def relative_strength_idx(self, df, n=14):
        self.df = df
        self.n = n
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=n).mean()
        avg_loss = pd.Series(loss).rolling(window=n).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value of 50

