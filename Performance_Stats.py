import numpy as np
import pandas as pd
import time
from datetime import datetime as dt


def total_return(return_col, annualized=True):
    r = (return_col + 1).cumprod()[-1] -1
    if annualized == True:
        days = (return_col.index[-1] - return_col.index[0]).days
        r = (r + 1) ** (1 / (days / 365)) - 1
    return r
        
def return_std(return_col, annualized = True):
    sd = return_col.std()
    if annualized:
        sd = return_col.std() * np.sqrt(251)
    return sd

def sharpe_ratio(return_col):
    """Returns annualized Sharpe Ratio"""
    ann_return = total_return(return_col, annualized=True)
    ann_std = return_std(return_col,annualized=True)
        
    return ann_return / ann_std

def sortino_ratio(return_col):
    """Returns annualized Sortino Ratio"""
    dwn_returns = return_col[return_col < 0]
    ann_return = total_return(return_col, annualized=True)
    ann_dwn_std = return_std(dwn_returns, annualized=True)
    
    return ann_return / ann_dwn_std

def max_drawdown(return_col):
    """Returns MDD over column"""
    # source of code: https://quant.stackexchange.com/questions/57703/implementation-of-maximum-drawdown-in-python-working-directly-with-returns
    
    cum_rets = (1 + return_col).cumprod() - 1
    nav = ((1 + cum_rets) * 100).fillna(100)
    hwm = nav.cummax()
    dd = nav / hwm - 1

    return min(dd)

def calmar_ratio(return_col):
    ann_return = total_return(return_col, annualized=True)
    mdd = max_drawdown(return_col)
    
    return ann_return / np.abs(mdd)

def compute_annual_stats(df, column, stat):
    start_yr = df.index[0].year
    end_yr = df.index[-1].year
    
    stats = []
    for yr in range(start_yr, end_yr+1):
        start = '01-01-' + str(yr)
        end = '12-31-' + str(yr)
        df_a = df.loc[start:end]
        stats.append(stat(df_a[column]))
    return stats



def turnover_cost(df_pred, tickers, bps=1):
    """Calculate costs from asset turnover"""
    df_pred = df_pred.sort_index(ascending=True)
    port_value = 1 # assume portfolio value = 1 after first time period
    costs = []
    cost_returns = []
    for i in range(df_pred.shape[0] - 1):
        cost_ticker = []
        for j, ticker in enumerate(tickers):
            asset_value_0 = port_value * df_pred[str(tickers[j]) + '_port_weight'][i]
            asset_value_1 = asset_value_0 * (1 + df_pred['return_next_' + str(tickers[j])][i])
            port_value_next = port_value * (1+df_pred['return_next_portfolio'][i])
            asset_value_next =  port_value_next * df_pred[str(tickers[j]) + '_port_weight'][i+1]        
            val_change = np.abs((asset_value_next - asset_value_1))
            cost = val_change * bps / 10000 * -1
            cost_ticker.append(cost)
        cost_sum = np.sum(cost_ticker)
        costs.append(cost_sum)
        cost_return = cost_sum / port_value
        cost_returns.append(cost_return)
        port_value *= (1 + df_pred['return_next_portfolio'][i] + cost_return)
    cost_returns.append(0)
    df_pred['cost_return_next'] = cost_returns
    df_pred['cost_return'] = df_pred['cost_return_next'].shift(1)
    df_pred['total_portfolio_return'] = df_pred['cost_return'] + df_pred['return_portfolio']
    return df_pred

def turnover(df_pred, tickers):
    """Determine daily asset turnover"""
    port_value = 1
    
    # id return col
    for c in df_pred.columns:
        if 'return_portfolio' in c:
            return_col = c
 
    turnover = []
    for i in range(df_pred.shape[0] - 1):
        turnover_ticker = []
        for j, ticker in enumerate(tickers):
            asset_value_0 = port_value * df_pred[str(tickers[j]) + '_port_weight'][i]
            asset_value_1 = asset_value_0 * (1 + df_pred['IvMean60_return_next_' + str(tickers[j])][i])
            port_value_next = port_value * (1+df_pred['return_next_portfolio'][i])
            asset_value_next =  port_value_next * df_pred[str(tickers[j]) + '_port_weight'][i+1]        
            val_change = np.abs((asset_value_next - asset_value_1))
            turnover_ticker.append(val_change)
        turnover_sum = np.sum(turnover_ticker)
        turnover.append(turnover_sum)
        port_value *= (1 + df_pred['return_next_portfolio'])
    turnover.append(0)
    df_pred['turnover'] = turnover
    return df_pred


def backtest(port_weights, returns, n=10, model='attn'):
    
    if model not in ['attn', 'gru', 'lstm']:
        print('Value Error: Invalid model selection, use attn, gru or lstm')
        sys.exit()
        
    port_return_adj = []
    for i in range(port_weights.shape[0] - 1):   # minus one because we have no returns in first period 
        x = port_weights.iloc[i].sort_values()
        net = x.sum()
        top_n_longs = x[-(int(n/2)):]
        new_long = ((.50 + (net / 2)) / top_n_longs.sum()) * top_n_longs
        top_n_shorts = x[:int((n/2))]
        new_short = ((-.50 + (net / 2)) / top_n_shorts.sum()) * top_n_shorts
        positions = new_long.append(new_short)
        
        ret = returns.iloc[i + 1]
        pos_return = []
        for p in positions.index:
            if model == 'gru':
                name = 'IvMean60_return_' + p[:-16]
            else:
                name = 'IvMean60_return_' + p[:-17]
            pos_return.append(positions[p] * ret[name])
        
        port_return_adj.append(np.sum(pos_return))
                
        
        
    port_return_adj_series = pd.Series(port_return_adj, index=port_weights.index[1:])
    
    
    return port_return_adj_series




            