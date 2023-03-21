import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import sys
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
from datetime import datetime as dt
from datetime import date
from plotly.subplots import make_subplots
from dash_bootstrap_templates import load_figure_template


doc_style = {'padding':'0px 60px'}

#style=doc_style
#### Documentation Start ####
doc = Dash(external_stylesheets=[dbc.themes.LUMEN])

load_figure_template('LUMEN')

doc.layout = html.Div(children=[
    dbc.Row([
        dbc.Col(style={'background-color': 'midnightblue'}),
        
        dbc.Col(html.H1('Implied Volatility Portfolio Optimization with Deep Learning',
                       style={"color": "white"}),
                width=10,
                style={'background-color': 'midnightblue'}),
        
        html.Br(),
    ]),
    
    dbc.Row([
        dbc.Col(style={'background-color': 'midnightblue'}),
        
        dbc.Col(html.H5('Visualization of UCLA MAS Program Thesis - Dylan Jorling',
                       style={"color": "white"}),
                width=10,
                style={'background-color': 'midnightblue'}),
        html.Br(),
    ]),
    dbc.Row([
        html.H2('Dashboard Documentation', style=doc_style),
        html.Br(),
    ]),
    dbc.Row([
        html.H4([html.B('General Dashboard Overview')]),
        html.Br(),
        
        html.P(['There are two main categories of data contained in this dashboard. The first category includes daily implied',
                'volatility portfolio weights output by the three trained machine learning models used in the study. An equal',
                '-weighted portfolio option is also often displayed to compare model performances to an untrained baseline.',
                'The second category includes 60-day until expiration implied volatility data for the 315 stocks and ETFs used', 
                'to train the machine learning models. Note that all data labeled “implied volatility” references this',  
                'approximately 60-day until expiration data.'],),
        
        html.Br(),],
        style=doc_style),
    
    dbc.Row([
        html.H4([html.B('Model Metrics')]),
        html.Br(),
        html.H5('Rolling Parameters by Model'),
        html.P([html.I('Adjustable Parameters: Performance Metric, Date Range')]),
        html.P([html.Li('This plot shows annualized rolling financial metrics for each model, as well as the equal',
                        'weighted baseline. ')]),
        html.H5('Model Monthly Returns Heatmap'),
        html.P([html.I('Adjustable Parameters: Model')]),
        html.P(html.Li(['This plot displays monthly returns in percentage terms for the selected model over the research time ',
                        'frame of 2014 to 2022.'],))
    ],
    style=doc_style),
    
    
    dbc.Row([
        html.H4([html.B('Volatility Exposure and Long/Short IV Levels')]),
        html.P([html.I('Adjustable Parameters: Model, Rolling Window Type, Date Range')]),
        html.P(html.Li(['This is a dual y-axis plot displaying net rolling implied volatility exposure of each model over ',
                        'time (y=-axis on left). Net implied volatility exposure can be defined as:'],)),
        
        html.P(html.Li(['Net IV exposure = |Portfolio Weights of all Long Positions| - |Portfolio Weights of all Short',
                        'Positions|'],)),
        
        html.P(html.Li(['The mean implied volatility exposure throughout the selected timeframe is also displayed as a ',
                        'vertical horizontal line. The other two lines, plotted against the Implied Volatility y-axis on the ',
                        'right, show the mean raw implied volatility average of all long positions (in green) and all short ',
                        'positions (in red).'],)),
        
        html.Br(),],

        style=doc_style),
    
    dbc.Row([
        html.H4([html.B('Asset IV over time and Correlations')]),
        html.P([html.I('Adjustable Parameters: Assets, Years (slider on bottom of panel)')]),
        html.P(html.Li(['These two plots are meant to go together to analyze trends in implied volatility and correlations',
                       ' between the implied volatility of assets.'],)),
        html.H5('Rolling Implied Volatility'),
        html.P([html.I('Additional Adjustable Parameters: Rolling Window Type')]),
        html.P([html.Li('This plot displays rolling IV for selected assets for selected years.')]),
        html.H5('Implied Volatility Correlation Heatmap'),
        html.P([html.Li('This plot displays a heatmap of implied volatility correlations for selected assets in the',
                        'given timeframe.')]),
        
        html.Br(),],

        style=doc_style),
    
    dbc.Row([
        html.H4([html.B('Model Comparison')]),
        html.H5('Portfolio Values'),
        html.P([html.I('Adjustable Parameters: Date Range')]),
        html.P(html.Li(['This plot displays cumulative portfolio values throughout time for each model as well as the ',
                        'baseline. Portfolio value is assumed to be 1 at the beginning of the selected timeframe.'])),
        html.H5('Model Annual Returns'),
        html.P([html.I('Adjustable Parameters: Model, Years (Slider beneath plots)')]),
        html.P(html.Li('This plot displays annual returns for the selected model and timeframe. ')),
        
        html.Br(),],

        style=doc_style),
    
    dbc.Row([
        html.H4([html.B('Plot Interactivity')]),
        html.H5([html.Li('Hoverability: Hover over plots for information about specific points')]),
        html.H5([html.Li('Zoomability: Click and drag on any plot to zoom in; double click to zoom out.')]),
        html.H5([html.Li('Display Options: For line plots, toggle labels in legend to hide/display certain lines.')]),

        html.Br(),],

        style=doc_style),   
    
]) ## final container brackets do not move
                      
      

if __name__ == '__main__':
    doc.run_server(debug=False)
        
        
        
        
        
        
        
        
        
        
        
        
    
