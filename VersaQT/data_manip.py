from typing import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm



# Bars =================================
def getBar(data: pd.DataFrame, threshold: int, kind: str = "tick", volume_col: str = "Volume", dollar_col: str = "Close"):
    data_ = data.copy()

    if (kind == "volume") and (volume_col not in data.columns):
        raise ValueError("Invalid volume_col name")
    elif (kind == "dollar"):
        if (volume_col not in data.columns) or (dollar_col not in data.columns):
            raise ValueError("Invalid volume_col or dollar_col name")
        
    if kind not in ["tick", "volume", "dollar"]:
        raise ValueError("Invalid kind option. Available options are: tick, volume and dollar")

    if kind == "tick":
        data_["TickCount"] = 1  # Each row is a tick
        col_name = "TickCount"
    elif kind == "volume":
        col_name = volume_col
    elif kind == "dollar":
        data_["DollarVolume"] = data[dollar_col] * data[volume_col]
        col_name = "DollarVolume"


    data_["CummTicks"] = data_[col_name].cumsum()    
    data_["BarId"] = data_["CummTicks"] // threshold

    data_ = data_.groupby("BarId").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
        "Date": ["first", "last"]
    })

    data_.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'StartTime', 'EndTime']

    return data_


# Fractional Differentiation

class FractionalDifferentitation():

    def fit(self, data, column):
        self.data = data
        self.column=column

        self.d = None
        self.MFD_d = None
        

    @staticmethod
    def __getWeights_FFD(d, thres):
        # Fixed Frac Diff
        # thres>0 drops insignificant weights
        w = [1.]
        k = 1
        while abs(w[-1]) >= thres:  
            w_ = -w[-1] / k * (d - k + 1)
            w.append(w_)
            k += 1
        w = np.array(w[ : : -1]).reshape(-1, 1)[1 : ]  
        return w

    def fracDiff_FFD(self, d=None, thres = 1e-5, start=0, end=1, interval=10, data=None):
        """
        Constant width window (new solution)
        Note 1: thres determines the cut-off weight for the window
        Note 2: d can be any positive fractional, not necessarily bounded [0,1].
        """

        if data is None:
            data = self.data

        if (d is None) and (self.d is None):
            print("Calculating optimal differentiation degree")
            self.d = self.OptimalFFD(start, end, interval, thres)
            print(f"Optimal d found as {self.d}")
        
        if d is None:
            d = self.d

        w = self.__getWeights_FFD(d, thres)

        width = len(w) - 1

        df = {} 
        for name in data.columns:
            seriesF = data[[name]].ffill().dropna()
            df_ = pd.Series() 
            for iloc1 in range(width, seriesF.shape[0]):
                loc0 = seriesF.index[iloc1 - width]
                loc1 = seriesF.index[iloc1]
                if not np.isfinite(data.loc[loc1,name]):
                    continue # exclude NAs
                
                df_[loc1] = np.dot(w.T, seriesF.loc[loc0 : loc1])[0, 0]
                
            df[name] = df_.copy(deep = True)
        df = pd.concat(df, axis = 1)
        return df
    
    def OptimalFFD(self, start = 0, end = 1, interval = 10, t=1e-2):
    
        data = self.data[self.column]

        for d in np.linspace(start, end, interval):    
            dfx = self.fracDiff_FFD(d, thres = t, data=data.to_frame())
            column = dfx.columns[0]
            if sm.tsa.stattools.adfuller(dfx[column], maxlag=1,regression='c',autolag=None)[1] < 0.05:
                self.d = d
                return d
        raise ValueError('no optimal d')
    
    @staticmethod
    def __getWeightsMFD(d, size, break_zero=False):
        """
            d: Degree of differentation (integer or float)
            size: Max iteration - summation to infinity
        """

        w=[1.]

        for k in range(1, size):
            w_ = -w[-1]*(d-k+1)/k
            w.append(w_)

            if break_zero and w_ == 0:
                break
        
        return np.array(w[::-1]).reshape(-1, 1)
    
    def fracDiff_MFD(self, d=None, thres = .01, start=0, end=1, interval=10, data=None):
        """
        Increasing width window, with treatment of NaNs (Standard Fracdiff, expanding window)
        Note 1: For thres=1, nothing is skipped.
        Note 2: d can be any positive fractional, not necessarily bounded [0,1].
        """

        if data is None:
            data = self.data
        
        if (d is None) and (self.MFD_d is None):
            print("Calculating optimal differentiation degree")
            self.MFD_d = self.OptimalMFD(start, end, interval, thres)
            print(f"Optimal d found as {self.MFD_d}")

        if d is None:
            d = self.MFD_d
        
        
        w = self.__getWeights(d, data.shape[0]) 
        w_ = np.cumsum(abs(w)) 
        w_ /= w_[-1] #
        skip = w_[w_ > thres].shape[0] 

        df = {} 
        for name in data.columns:
            # fill the na prices
            seriesF = data[[name]].ffill().dropna()
            df_ = pd.Series() 
            for iloc in range(skip, seriesF.shape[0]):
                loc = seriesF.index[iloc]
                
                test_val = data.loc[loc,name] 
                if isinstance(test_val, (pd.Series, pd.DataFrame)):
                    test_val = test_val.resample('1m').mean()
                
                if not np.isfinite(test_val).any():
                    continue # exclude NAs
                try: # the (iloc)^th obs will use all the weights from the start to the (iloc)^th
                    df_.loc[loc]=np.dot(w[-(iloc+1):,:].T, seriesF.loc[:loc])[0,0]
                except:
                    continue
            df[name] = df_.copy(deep = True)
        df = pd.concat(df, axis = 1)
        return df
    
    def OptimalMFD(self, start = 0, end = 1, interval = 10, t=0.01):
        
        data = self.data[self.column]

        for d in np.linspace(start, end, interval):    
            dfx = self.fracDiff_MFD(d, thres = t, data=data.to_frame())
            if sm.tsa.stattools.adfuller(dfx[self.column], maxlag=1,regression='c',autolag=None)[1] < 0.05:
                self.MFD_d = d
                return d
        print('no optimal d')
        return d

    @staticmethod
    def __get_weight_loss(d, size):
        w=[1.]

        for k in range(1, size):
            w_ = -w[-1]*(d-k+1)/k
            w.append(w_)

        w_total = np.sum(np.abs(w))

        w_loss = []
        for k in range(len(w)):
            w_loss_ = np.sum(np.abs(w[k:]))/w_total
            w_loss.append(w_loss_)

        # sns.lineplot(x=[k for k in range(len(w))], y=w_loss)

        return w_loss
