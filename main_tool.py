import os
import cv2
import json
import parse
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, callback
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

#from utils.general import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='point')
parser.add_argument('--host', type=str, default='127.0.0.1')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--file', type=str, default=False)
parser.add_argument('--old', type=str, default= False)
parser.add_argument('--new', type=str, default= False)
args = parser.parse_args()

mode = args.mode
host = args.host
debug = args.debug
eval_file_list = []

app = dash.Dash(__name__)



old_preds = pd.read_csv(f'{args.old}')
new_preds = pd.read_csv(f'{args.new}')

diff_pers = f"{(new_preds['Visibility'].sum() - old_preds['Visibility'].sum())/old_preds['Visibility'].sum() * 100:.1f}%"
if new_preds['Visibility'].sum() - old_preds['Visibility'].sum() > 0:
    diff_pers = "+" + diff_pers + " 📈"
else:
    diff_pers = "-" + diff_pers + " 📉"
visibility_diff = f"{new_preds['Visibility'].sum() - old_preds['Visibility'].sum()} / {diff_pers}"

app.layout = html.Div(children=[
    # Labels lists
    html.Div(children=[
        html.Div(children=[
            html.Label([f'Old CSV: {args.old}'], style={'font-weight': 'bold', "text-align": "center"}),
            #dcc.Dropdown(eval_file_list, eval_file_list[0]['value'], id='eval-file-1-dropdown')
        ], style=dict(width='20%', margin='10px')),
        html.Div(children=[
            html.Label([f'New CSV: {args.new}'], style={'font-weight': 'bold', "text-align": "center"}),
            #dcc.Dropdown(eval_file_list, eval_file_list[0]['value'], id='eval-file-2-dropdown')
        ], style=dict(width='20%', margin='10px')),
        html.Div(children=[
            html.Label([f'Video: {args.file}'], style={'font-weight': 'bold', "text-align": "center"}),
        ], style=dict(width='20%', margin='10px'))
    ], style={'display':'flex', 'justify-content':'center', 'text-align':'center'}),
    # Metrics calculations
    html.Div(children=[
        html.H2([f'Visibility:'])
    ]),
    html.Div(children=[
        html.H2([f'OLD - {old_preds["Visibility"].sum()}'])
    ]),
    html.Div(children=[
        html.H2([f'NEW - {new_preds["Visibility"].sum()}'])
    ]),
    html.Div(children=[
        html.H2([f'Diff - {visibility_diff}'])
    ]),
    # Time series plot
    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='time_fig',
                figure=go.Figure().set_subplots(rows=2, cols=1),
                config={'scrollZoom':True}
            ),
        ], style=dict(width='90%')),
    ], style={'display':'flex', 'justify-content':'center', 'text-align':'center'}),
    # Frame plot
    html.Div(children=[
        dcc.Graph(
            id='frame_fig',
            figure=go.Figure(layout={'title': {'text':'Frame ID: 0'}}),
            config={'scrollZoom':True},
        ),
    ], style={'display':'flex', 'justify-content':'center', 'align-items': 'center'}, id='frameImg'),
    html.Div(children=[
        dcc.Slider(0, old_preds['Frame'].values[-1], 1, value=0, marks=None, id='framePicker',
                    tooltip={"placement": "top", "always_visible": True}
        ),
        html.Div(id='sliderFramePicker')
    ])
])

@callback(
    Output('sliderFramePicker', 'children'),
    Input('framePicker', 'value'))
def update_slider(value):
    update_frame(value)
    return

@callback(
    Output('frame_fig', 'figure'),
    [Input('framePicker','value'),]
)
def update_frame(frameId):
    reader = cv2.VideoCapture()
    reader.open(f'{args.file}')
    reader.set(1,frameId)
    ret, frame = reader.read()

    if old_preds.loc[frameId]['Visibility'] == 0:
        cv2.putText(frame, "NO BALL", (30, 30), fontFace=5, color= (0,0,255), fontScale= 1)
    else:
        cv2.circle(frame, (old_preds.loc[frameId]['X'], old_preds.loc[frameId]['Y']), radius= 5, color= (0,0,255), thickness= 2)    
    if new_preds.loc[frameId]['Visibility'] == 0:
        cv2.putText(frame, "NO BALL", (30, 50), fontFace=5, color= (0,255,0), fontScale= 1)
    else:
        cv2.circle(frame, (new_preds.loc[frameId]['X'], new_preds.loc[frameId]['Y']), radius= 7, color= (0,255,0), thickness= 2)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_fig = px.imshow(frame)
    frame_fig = go.Figure()
    frame_fig.add_trace(img_fig.data[0])
    frame_fig.update_layout(
                            width=1280,
                            height=720,
                            margin=dict(l=0, r=0, t=100, b=0),
                            title= f'Frame ID: {frameId+1}'
                            )
    return frame_fig


def Euclidean_Dist(df1, df2, cols=['x_coord','y_coord']):
    # calc euclidian dist only on both visible
    df1_cp = df1.copy()
    df2_cp = df2.copy()

    df1_cp.loc[df2_cp['Visibility'] == 0, 'X'] = 0
    df1_cp.loc[df2_cp['Visibility'] == 0, 'Y'] = 0

    df2_cp.loc[df1_cp['Visibility'] == 0, 'X'] = 0
    df2_cp.loc[df1_cp['Visibility'] == 0, 'Y'] = 0
    
    
    return np.linalg.norm(df1_cp[cols].values - df2_cp[cols].values,
                   axis=1)

@callback(
    Output('time_fig', 'figure'),
    Input('framePicker','value')
)
def calc_visability(frameId):
    time_fig = go.Figure().set_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.15, subplot_titles=('Visibility OLD', 'Visibility NEW', 'Diff as Euclidian distance from predictions (only on both visible)'))
    time_fig.add_trace(
                go.Bar(x=old_preds['Frame'], y=old_preds['Visibility'], name= 'old'), row=1, col=1)
    time_fig.add_trace(
        go.Bar(x=new_preds['Frame'], y=new_preds['Visibility'], name= 'new'),row=2, col=1)

    diff = Euclidean_Dist(old_preds, new_preds, cols= ['X','Y'])
    time_fig.add_trace(go.Scatter(x= old_preds['Frame'], y= diff, name='distance'), row=3, col=1)
    return time_fig


if __name__ == '__main__':
   app.run_server(host='127.0.0.1', debug=True, use_reloader=False)