import logging

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from safeskijump.functions import make_jump

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BS_URL = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'

app = dash.Dash()
app.css.append_css({'external_url': BS_URL})

start_pos_widget = html.Div([
    html.P('Start Position [m]'),
    dcc.Input(id='start_pos',
              placeholder='Start Position [meters]',
              inputmode='numeric',
              type='number',
              value=10.0,
              min=0.0,
              step=1.0)
    ])

approach_len_widget = html.Div([
    html.P('Approach Length [m]'),
    dcc.Input(id='approach_len',
              placeholder='Approach Length [meters]',
              inputmode='numeric',
              type='number',
              value=50.0,
              min=1.0,
              step=1.0)
    ])

fall_height_widget = html.Div([
    html.P('Fall Height [m]'),
    dcc.Input(id='fall_height',
              placeholder='Fall Height [meters]',
              inputmode='numeric',
              type='number',
              value=0.2,
              max=3.0,
              min=0.1,
              step=0.1,
              )
    ])

slope_angle_widget = html.Div([
    html.P('Slope Angle: 10 degrees', id='slope-text'),
    dcc.Slider(
        id='slope_angle',
        min=0,
        max=45,
        step=1,
        value=10,
        marks={0: '0 [deg]', 45: '45 [deg]'},
        # updatemode='drag'
        )
    ])

takeoff_angle_widget = html.Div([
    html.P('Takeoff Angle: 10 degrees', id='takeoff-text'),
    dcc.Slider(
        id='takeoff_angle',
        min=0,
        max=45,
        step=1,
        value=20,
        marks={0: '0 [deg]', 45: '45 [deg]'},
        # updatemode='drag'
        )
    ])

layout = go.Layout(autosize=False,
                   width=1200,
                   height=800,
                   yaxis={'scaleanchor': 'x'})  # equal aspect ratio

# TODO : See if the className can be added to Graph instead of Div.
graph_widget = html.Div([dcc.Graph(id='my-graph',
                                   figure=go.Figure(layout=layout))],
                        className='col-md-12')

row1 = html.Div([html.H1('Ski Jump Design')], className='row')

row2 = html.Div([graph_widget], className='row')

row3 = html.Div([html.P('Invalid Jump Design')], id='error-bar',
                className='alert alert-warning', style={'display': 'none'})

row4 = html.Div([html.Div([start_pos_widget], className='col-md-4'),
                 html.Div([approach_len_widget], className='col-md-4'),
                 html.Div([fall_height_widget], className='col-md-4'),
                 ], className='row')

row5 = html.Div([html.Div([slope_angle_widget], className='col-md-6'),
                 html.Div([takeoff_angle_widget], className='col-md-6'),
                 ], className='row')

app.layout = html.Div([row1, row2, row3, row4, row5], className='container')


@app.callback(Output('slope-text', 'children'),
              [Input('slope_angle', 'value')])
def update_slope_text(slope_angle):
    slope_angle = float(slope_angle)
    return 'Slope Angle: {:0.0f} degrees'.format(slope_angle)


@app.callback(Output('takeoff-text', 'children'),
              [Input('takeoff_angle', 'value')])
def update_takeoff_text(takeoff_angle):
    takeoff_angle = float(takeoff_angle)
    return 'Takeoff Angle: {:0.0f} degrees'.format(takeoff_angle)


inputs = [Input('slope_angle', 'value'),
          Input('start_pos', 'value'),
          Input('approach_len', 'value'),
          Input('takeoff_angle', 'value'),
          Input('fall_height', 'value')
          ]


@app.callback(Output('my-graph', 'figure'), inputs)
def update_graph(slope_angle, start_pos, approach_len, takeoff_angle,
                 fall_height):

    slope_angle = -float(slope_angle)
    start_pos = float(start_pos)
    approach_len = float(approach_len)
    takeoff_angle = float(takeoff_angle)
    fall_height = float(fall_height)

    logging.info('Calling make_jump({}, {}, {}, {}, {})'.format(
        slope_angle, start_pos, approach_len, takeoff_angle, fall_height))

    surfs = make_jump(slope_angle, start_pos, approach_len, takeoff_angle,
                      fall_height)
    slope, approach, takeoff, landing, trans, flight = surfs

    return {'data': [
                     {'x': slope.x, 'y': slope.y, 'name': 'Slope',
                      'line': {'color': 'black', 'dash': 'dash'}},
                     {'x': approach.x, 'y': approach.y, 'name': 'Approach',
                      'line': {'width': 4}},
                     {'x': takeoff.x, 'y': takeoff.y, 'name': 'Takeoff',
                      'line': {'width': 4}},
                     {'x': landing.x, 'y': landing.y, 'name': 'Landing',
                      'line': {'width': 4}},
                     {'x': trans.x, 'y': trans.y, 'name': 'Landing Transition',
                      'line': {'width': 4}},
                     {'x': flight.x, 'y': flight.y, 'name': 'Flight',
                      'line': {'dash': 'dot'}},
                    ],
            'layout': layout}

if __name__ == '__main__':
    app.run_server()
