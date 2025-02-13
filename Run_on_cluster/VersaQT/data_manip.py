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
def __getWeights(d, size, break_zero=False):
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


def __plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = __getWeights(d, size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')

    sns.lineplot(w)


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

def fracDiff(series, d, thres = .01):
    """
    Increasing width window, with treatment of NaNs (Standard Fracdiff, expanding window)
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    #1) Compute weights for the longest series
    w = __getWeights(d, series.shape[0]) # each obs has a weight
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w)) # cumulative weights
    w_ /= w_[-1] # determine the relative weight-loss
    skip = w_[w_ > thres].shape[0]  # the no. of results where the weight-loss is beyond the acceptable value
    #3) Apply weights to values
    df = {} # empty dictionary
    for name in series.columns:
        # fill the na prices
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series() # create a pd series
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc] # find the iloc th obs 
            
            test_val = series.loc[loc,name] # must resample if duplicate index
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

def fracDiff_FFD(series, d, thres = 1e-5):
    """
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    #1) Compute weights for the longest series
    w = __getWeights_FFD(d, thres)
    # w = getWeights(d, series.shape[0])
    #w=getWeights_FFD(d,thres)
    width = len(w) - 1
    #2) Apply weights to values
    df = {} # empty dict
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series() # empty pd.series
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):
                continue # exclude NAs
            #try: # the (iloc)^th obs will use all the weights from the start to the (iloc)^th
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0 : loc1])[0, 0]
            # except:
            #     continue
            
        df[name] = df_.copy(deep = True)
    df = pd.concat(df, axis = 1)
    return df

def plotMinFFD(instName):
    from statsmodels.tsa.stattools import adfuller

    if isinstance(instName, pd.DataFrame):
        instName.to_csv("temp_df.csv")
        instName = "temp_df"

    path = ''
    out = pd.DataFrame(columns= ['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    df0 = pd.read_csv(path + instName +'.csv',index_col = 0, parse_dates = True)
    for d in np.linspace(0, 1, 11):
        df1 = np.log(df0[['Close']]).resample('1D').last() # downcast to daily obs
        df2 = fracDiff_FFD(df1, d, thres = .01)
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
        df2 = adfuller(df2['Close'], maxlag = 1, regression = 'c', autolag = None)
        out.loc[d] = list(df2[ : 4]) + [df2[4]['5%']] + [corr] # with critical value
    out.to_csv(path + instName + '_testMinFFD.csv')
    out[['adfStat', 'corr']].plot(secondary_y = 'adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth = 1, color = 'r', linestyle = 'dotted')
    # plt.savefig(path + instName + '_testMinFFD.png')
    return

def plotMinMFD(instName):
    from statsmodels.tsa.stattools import adfuller

    if isinstance(instName, pd.DataFrame):
        instName.to_csv("temp_df.csv")
        instName = "temp_df"

    path = ''
    out = pd.DataFrame(columns= ['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    df0 = pd.read_csv(path + instName +'.csv',index_col = 0, parse_dates = True)
    for d in np.linspace(0, 1, 11):
        df1 = np.log(df0[['Close']]).resample('1D').last() # downcast to daily obs
        df2 = fracDiff(df1, d, thres = .01)
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
        df2 = adfuller(df2['Close'], maxlag = 1, regression = 'c', autolag = None)
        out.loc[d] = list(df2[ : 4]) + [df2[4]['5%']] + [corr] # with critical value
    out.to_csv(path + instName + '_testMinFFD.csv')
    out[['adfStat', 'corr']].plot(secondary_y = 'adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth = 1, color = 'r', linestyle = 'dotted')
    # plt.savefig(path + instName + '_testMinFFD.png')
    return

def OptimalMFD(data, start = 0, end = 1, interval = 10, t=0.01, column="price"):
    
    for d in np.linspace(start, end, interval):    
        dfx = fracDiff(data.to_frame(), d, thres = t)
        if sm.tsa.stattools.adfuller(dfx[column], maxlag=1,regression='c',autolag=None)[1] < 0.05:
            return d
    print('no optimal d')
    return d

def OptimalFFD(data, start = 0, end = 1, interval = 10, t=1e-5, column="price"):
    
    for d in np.linspace(start, end, interval):    
        dfx = fracDiff_FFD(data.to_frame(), d, thres = t)
        if sm.tsa.stattools.adfuller(dfx[column], maxlag=1,regression='c',autolag=None)[1] < 0.05:
            return d
    print('no optimal d')
    return d