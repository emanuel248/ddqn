import os
import copy
import time
import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from flask_caching import Cache


external_stylesheets = [
    # Dash CSS
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    # Loading screen CSS
    'https://codepen.io/chriddyp/pen/brPBPO.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache'
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

N = 100

aapl_df = pd.read_csv('tsla.us.txt')


app.layout = html.Div([
    html.Div([
        html.Div(dcc.Graph(id='graph-1'), className="twelve columns"),
    ], className="row"),
    html.Div(style={'height': '10px', 'background': '#f1f1f1'}),
    html.Div([
        html.Div(dcc.Graph(id='graph-2'), className="twelve columns"),
    ], className="row"),

    # hidden signal value
    html.Div(id='signal', style={'display': 'none'})
])

# perform expensive computations in this "global store"
# these computations are cached in a globally available
# redis memory store which is available across processes
# and for all time.
@cache.memoize()
def global_store(value):
    # simulate expensive query
    print('Computing value with {}'.format(value))
    time.sleep(5)
    return aapl_df


def generate_figure(value, figure):
    fig = copy.deepcopy(figure)
    filtered_dataframe = global_store(value)
    fig['data'][0]['x'] = filtered_dataframe['Date']
    fig['data'][0]['y'] = filtered_dataframe['Close']
    fig['data'][0]['color'] = filtered_dataframe['Open']
    fig['layout'] = {'margin': {'l': 20, 'r': 10, 'b': 20, 't': 10}}
    return fig 


@app.callback(Output('graph-1', 'figure'), [Input('signal', 'children')])
def update_graph_1(value):
    # generate_figure gets data from `global_store`.
    # the data in `global_store` has already been computed
    # by the `compute_value` callback and the result is stored
    # in the global redis cached
    return generate_figure('oranges', {
        'data': [{
            'type': 'scatter',
            'mode': 'lines+markers',
            'line': {'shape': 'spline', 'width': 0.7},
            'marker': {'symbol':'circle', 'size': 3}
        }]
    })


@app.callback(Output('graph-2', 'figure'), [Input('signal', 'children')])
def update_graph_2(value):
    return generate_figure('apples', {
        'data': [{
            'type': 'scatter',
            'mode': 'lines+markers',
            'line': {'shape': 'spline', 'width': 0.7},
            'marker': {'symbol':'circle', 'size': 3}
        }]
    })


if __name__ == '__main__':
    app.run_server(debug=True, processes=1)