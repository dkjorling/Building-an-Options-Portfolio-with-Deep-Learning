import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import warnings
import seaborn as sns
import os
import plotly.offline as pyo
import sys
from plotly.subplots import make_subplots

# .py file imports
import Performance_Stats as ps

warnings.filterwarnings('ignore')
pyo.init_notebook_mode()

def my_style(fig, lines=True):
    fig.update_layout(plot_bgcolor='white')
    
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey')
    
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey')
    
    fig.update_layout(title_x=0.5, title_y=0.88)
    
    return fig

def midnightblue(fig, lines=True):
    
    fig.update_layout(
        font_color="midnightblue",
        title_font_color="midnightblue",
        legend_title_font_color="midnightblue")
    
    return fig

def cname(return_col):
    if 'return_portfolio' in return_col.name:
        name = return_col.name[17:].upper()
    elif return_col.name == 'return_benchmark':
        name = 'EW Baseline'
    else:
        name = return_col.name[16:]
    
    return name
            
    
def plot_cum_return(return_cols, title, start=None, end=None):
    """Takes a list of return series and plots cumulative returns on a single plot.
    
    Keyword arguments:
    start: date string in 'DD-mm-YYYY' format that indicates start of plot (default=None)
    end: date string in 'DD-mm-YYYY' format that indicates start of plot (default=None)
    
    """
                    
    if start == None:
        start=return_cols[0].index[0]
    if end == None:
        end=return_cols[0].index[-1]
                    

    layout = go.Layout(title=title,
                      yaxis=dict(title='Portfolio Value'),
                      xaxis=dict(title='Year'))
    
    fig = go.Figure(layout=layout)
    for col in return_cols:
        col = col.loc[start:end]
        idx = col.index
        col = col+1
        
        # set name in legend
        name = cname(col)
        
        fig.add_trace(go.Scatter(x=idx,
                                y=col.cumprod(),
                                mode='lines',
                                name=name))

    midnightblue(fig)

    
    return fig


def plot_rolling_metric(return_cols, metric, title, start=None, end=None):
    """Takes list of return series and plots 1-year rolling metric beginning 6 months into series.
    
    Keyword arguments:
    start: date string in 'DD-MM-YYYY' format that indicates start of plot (default=None)
    end: date string in 'DD-MM-YYYY' format that indicates start of plot (default=None)
    
    """
    
    if start == None:
        start=return_cols[0].index[0]
    if end == None:
        end=return_cols[0].index[-1]
        
    layout = go.Layout(title=title, xaxis=dict(title='Year'))

    
    fig = go.Figure(layout=layout)
    for col in return_cols:
        col = col.loc[start:end]
        col.iloc[0] = np.nan
        col = col.rolling('365d').apply(metric)[151:]
        idx = col.index
        
        # set name in legend
        name = cname(col)
        
        fig.add_trace(go.Scatter(x=idx,
                                y=col,
                                mode='lines',
                                name=name))
        
    midnightblue(fig)

    
    return fig



def return_matrix(return_col, start_yr=2014, end_yr=2022):
    s = '01-01-' + str(start_yr)
    e = '12-31-' + str(end_yr)
    return_col = (100 * return_col.loc[s:e])
    monthly = list(return_col.resample('M').sum().round(2))
    years = list(range(start_yr, end_yr +1))
    months = ['Jan', 'Feb', 'Mar','Apr',
              'May', 'Jun', 'Jul','Aug',
              'Sep', 'Oct', 'Nov', 'Dec']
    
    returns = []
    i = 0
    
    for j in range(len(years)):
        returns.append(monthly[i:(i+12)])
        i += 12
    
    
    name = cname(return_col)
    title = name + ' Monthly Returns in %'
    layout = go.Layout(title=title,
                       xaxis=dict(title='Month'),
                      yaxis=dict(title='Year',
                                autorange="reversed")) 
    
    fig = go.Figure(layout=layout)
    
    fig.add_trace(go.Heatmap(
        z = returns,
        x = months,
        y = years,
        texttemplate="%{z}",
    colorscale = 'RdYlGn'))
    
    midnightblue(fig)
#    fig.update_layout(title_x=0.5, title_y=0.88)
    fig.update_traces(showscale=False)
    
    return fig
        
    
    
def single_stat_bar_yearly(data, col, stat, title, start_yr=2014, end_yr=2022):
    s = '01-01-' + str(start_yr)
    e = '12-31-' + str(end_yr+1)
    d = data.loc[s:e]


    
    years = list(range(start_yr, end_yr +1))
    
    x = ps.compute_annual_stats(d, col, stat)
    
    layout = go.Layout(title=title,
                       yaxis=dict(title='Year',
                                  autorange="reversed"),
                       xaxis=dict(title='Return')) 
    
    fig = go.Figure(layout=layout)
    
    fig.add_trace(go.Bar(x=x,
                         y=years,
                        orientation='h',
                        marker=dict(
                            color='skyblue')))
    
    midnightblue(fig)
    
    return fig
    
    
    
def plot_iv_exposure_level(data, model, window='30d', start='01-01-2014', end='12-31-2022'):
    """Plot net IV exposure, w average IV, regular average IV as three lines plots
    
    Keyword arguments:
    model must be one of 'Attention Transformer', 'GRU', 'LSTM'
    
    """
    
    if model not in ['Attention Transformer', 'GRU', 'LSTM']:
        print('Value Error: Invalid model selection, use Attention Transformer/GRU/LSTM')
        
        sys.exit()
        
    data = data.loc[start:end]
    
    # create rolling avg net iv exposure
    idx = data.index
    if model == 'Attention Transformer':
        wt = data.iloc[:, 630:945]
    elif model == 'GRU':
        wt = data.iloc[:, 946:1261]
    else:
        wt = data.iloc[:, 1262:1577]
    
    net_w = wt.sum(axis=1).rolling(window).mean()
    
    av_wt = wt.sum(axis=1).mean()
    av_wt_s = pd.Series(np.repeat(av_wt, data.shape[0]), index=idx)
    
    # calc avg vol level
    ivs = data.iloc[:, :315]
    iv_avg = ivs.mean(axis=1).rolling(window).mean()
    
    # create rolling avg wIV exposure long/ short
    ivs = data.iloc[:, :315].values
    w_np = wt.values
    
    
    avg_neg_w_exp = []
    avg_pos_w_exp = []
    for i, w in enumerate(w_np):
        list_w = list(w)
    
        neg_idx = [j for j, x in enumerate(list_w) if x < 0]
        pos_idx = [j for j, x in enumerate(list_w) if x >= 0]
    
        neg_w = [w[j] for j in neg_idx]
        pos_w = [w[j] for j in pos_idx]
    
        neg_ivs = [ivs[i][j] for j in neg_idx]
        pos_ivs = [ivs[i][j] for j in pos_idx]
    
        scalar_n = 1 / np.array(neg_w).sum()
        neg_w_avg_iv = (np.array(neg_w) * np.array(neg_ivs)).sum() * scalar_n

        scalar_p = 1 / np.array(pos_w).sum()
        pos_w_avg_iv = (np.array(pos_w) * np.array(pos_ivs)).sum() * scalar_p
    
        avg_neg_w_exp.append(neg_w_avg_iv)
        avg_pos_w_exp.append(pos_w_avg_iv)
    
    neg_s = pd.Series(avg_neg_w_exp, index = idx).rolling(window).mean()
    pos_s = pd.Series(avg_pos_w_exp, index = idx).rolling(window).mean()
    
    # set up plot
    title = model + ' Rolling ' + window+ ' Avg IV Exposure and Level'
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    
    
    fig.add_trace(go.Scatter(x=idx,
                             y=net_w,
                            mode='lines',
                            name='Net IV Exposure'),
                  secondary_y=False)
    

    
    fig.add_trace(go.Scatter(x=idx,
                             y=neg_s,
                             mode='lines',
                             name='Avg IV of Short Positions'),
                  secondary_y=True)
    
    fig.add_trace(go.Scatter(x=idx,
                             y=pos_s,
                             mode='lines',
                             name='Avg IV of Long Positions'),
                  secondary_y=True)
    
    fig.add_trace(go.Scatter(x=idx,
                             y=av_wt_s,
                             mode='lines',
                             line={'dash': 'dash', 'color': 'orange'},
                             name='Mean IV Exposure'),
                  secondary_y=False)
    
    # Add figure title
    fig.update_layout(title_text=title)

    # Set y-axes titles
    fig.update_yaxes(title_text="IV Net Exposure", secondary_y=False)
    fig.update_yaxes(title_text="Implied Volatility", secondary_y=True)
    
    midnightblue(fig)
    
    return fig

def correlation_heat(data, tickers, start='01-01-2014', end='12-31-2022'):
    data = data.loc[start:end]
    keep_cols = []
    for ticker in tickers:
        keep_cols.append('IvMean60_' + ticker)
        
    data = data[keep_cols]
    cr = data.corr()
    cr.columns=tickers
    
    title = 'Implied Volatility Correlation Heatmap'   
    layout = go.Layout(title=title) 
    
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Heatmap(
                            x=cr.columns,
                            y=cr.columns,
                            z=cr.values.tolist(),
                            colorscale='Temps',
                            zmin=0.2,
                            zmax=1))
    midnightblue(fig)
    
    return fig


def plot_asset_iv(data, tickers, window='30d', start='01-01-2014', end='12-31-2022'):
    data = data.loc[start:end]
    keep_cols = []
    for ticker in tickers:
        keep_cols.append('IvMean60_' + ticker)
    data = data[keep_cols]
    idx = data.index
    
    title = "Rolling " + window + " Implied Volatility"
    layout = go.Layout(title=title, xaxis=dict(title='Year'))

    fig = go.Figure(layout=layout)
    for col in data.columns:
        data[col].iloc[0] = np.nan
        rolling = data[col].rolling(window).mean()
        
        fig.add_trace(go.Scatter(x=idx,
                                y=rolling,
                                mode='lines',
                                name=col[9:]))
        
    midnightblue(fig)
    
    return fig

    rolling = data[col].rolling(window).mean()
    





























    
