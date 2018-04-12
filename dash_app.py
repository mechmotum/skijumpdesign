import os
import logging

import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from safeskijump.functions import make_jump
from safeskijump.classes import InvalidJumpError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BS_URL = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'
CUS_URL = 'https://moorepants.info/misc/skijump.css'

app = dash.Dash(__name__)
app.css.append_css({'external_url': [BS_URL, CUS_URL]})
server = app.server
if 'ONHEROKU' in os.environ:
    import dash_auth
    auth = dash_auth.BasicAuth(app, [['skiteam', 'howhigh']])

start_pos_widget = html.Div([
    html.H3('Start Position [m]'),
    dcc.Input(id='start_pos',
              placeholder='Start Position [meters]',
              inputmode='numeric',
              type='number',
              value=0.0,
              min=0.0,
              step=5.0)
    ])

approach_len_widget = html.Div([
    html.H3('Approach Length [m]'),
    dcc.Input(id='approach_len',
              placeholder='Approach Length [meters]',
              inputmode='numeric',
              type='number',
              value=30.0,
              min=10.0,
              step=5.0)
    ])

fall_height_widget = html.Div([
    html.H3('Fall Height [m]'),
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
    html.H3('Slope Angle: 10 degrees', id='slope-text'),
    dcc.Slider(
        id='slope_angle',
        min=0,
        max=45,
        step=1,
        value=10,
        marks={0: '0 [deg]',
               15: '15 [deg]',
               30: '30 [deg]',
               45: '45 [deg]'},
        )
    ])

takeoff_angle_widget = html.Div([
    html.H3('Takeoff Angle: 10 degrees', id='takeoff-text'),
    dcc.Slider(
        id='takeoff_angle',
        min=0,
        max=45,
        step=1,
        value=20,
        marks={0: '0 [deg]',
               15: '15 [deg]',
               30: '30 [deg]',
               45: '45 [deg]'},
        )
    ])

layout = go.Layout(autosize=False,
                   width=1000,
                   height=600,
                   paper_bgcolor='rgba(96, 164, 255, 0.0)',
                   plot_bgcolor='rgba(255, 255, 255, 0.5)',
                   xaxis={'title': 'Distance [m]', 'zeroline': False},
                   yaxis={'scaleanchor': 'x',  # equal aspect ratio
                          'title': 'Height [m]', 'zeroline': False})

# TODO : See if the className can be added to Graph instead of Div.
graph_widget = html.Div([dcc.Graph(id='my-graph',
                                   figure=go.Figure(layout=layout))],
                        className='col-md-12')

row1 = html.Div([html.H1('Equivalent Fall Height Ski Jump Design Tool',
                         style={'text-align': 'center',
                                'padding-top': '20px',
                                'color': 'white'})],
                className='page-header',
                style={'height': '100px',
                       'margin-top': '-20px',
                       'background': 'rgba(128, 128, 128, 0.75)',
                       'border-bottom': '4px solid #eee'})

row2 = html.Div([graph_widget], className='row')

row3 = html.Div([html.H2('Messages'), html.P('', id='message-text')], id='error-bar',
                className='alert alert-warning',
                style={'display': 'none'}
                )

row4 = html.Div([html.Div([start_pos_widget], className='col-md-4'),
                 html.Div([approach_len_widget], className='col-md-4'),
                 html.Div([fall_height_widget], className='col-md-4'),
                 ], className='row')

row5 = html.Div([html.Div([slope_angle_widget], className='col-md-5'),
                 html.Div([], className='col-md-2'),
                 html.Div([takeoff_angle_widget], className='col-md-5'),
                 ], className='row', style={'margin-top': 15})

markdown_text = """\
# Instructions

- Select a parent slope angle to match the grade you plan to build the ski jump
  on.
- Set the length of the approach to be the maximum distance along the slope
  before the jump that a skier can traverse when starting from a stop.
- Set the desired takeoff angle of the ramp exit.
- Choose a desired equivalent fall height.
- Inspect and view the graph of the resulting jump design using the menu bar.

# Explanation

This tool allows you to design a ski jump for takeoff speeds up to a maximum
that ensures no the jumper will always impact the slope at the same speed one
would if dropped vertically from a desired fall height onto a flat surface.

## Inputs

- **Slope Angle**: The downward angle of the parent slope you wish to build
  the jump on.
- **Start Position**:  Distance down the slope where the skier starts skiing
  from. The skier starts skiing at 0 m/s from this location.
- **Approach Length**: The distance along the slope that the skier slides on
  to build up speed. The skier reaches a theoretical maximum speed at the end
  of the approach and the jump shape is designed around this maximum achievable
  speed.
- **Takeoff Angle**: The upward angle, relative to horizontal, that the end of
  the takeoff ramp is set to.
- **Fall Height**: The desired equivalent fall height for the jump design.

## Outputs

- **Takeoff Surface**: This curve is designed to give a smooth, constant
  acceleration transition from the parent slope to the takeoff ramp. The skier
  catches air at the end of the takeoff ramp.
- **Landing Surface**: This curve ensures that skiers launching at any speeds
  from 0 m/s to the maximum achievable speed at the end of the approach always
  impacts the landing surface with a speed no greater than the impact speed
  from and equivalent vertical fall height.
- **Landing Transition Surface**: This surface ensures a smooth transition from
  the landing surface back to the parent surface.
- **Flight Trajectory**: This shows the flight path from the maximum achievable
  takeoff speed.

# Colophon

This website was designed by Jason K. Moore and Mont Hubbard and based based on
the work detailed in:

Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. "A Design
Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall Height."
Sports Engineering 18, no. 4 (December 2015): 227–39.
[https://doi.org/10.1007/s12283-015-0182-6](https://doi.org/10.1007/s12283-015-0182-6)."""

row6 = html.Div([dcc.Markdown(markdown_text)],
                className='row',
                style={'background-color': 'rgba(128, 128, 128, 0.9)',
                       'color': 'white',
                       'padding-right': '20px',
                       'padding-left': '20px',
                       'margin-top': '40px',
                       'text-shadow': '1px 1px black'})

app.layout = html.Div([row1, html.Div([row2, row3, row4, row5, row6],
                      className='container')])


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

nan_line = [np.nan]
blank_graph = {'data': [
                     {'x': [0.0, 0.0], 'y': [0.0, 0.0], 'name': 'Parent Slope',
                      'text': ['Invalid Jump Parameters'],
                      'mode': 'markers+text',
                      'textfont': {'size': 24},
                      'textposition': 'top',
                      'line': {'color': 'black', 'dash': 'dash'}},
                     {'x': nan_line, 'y': nan_line, 'name': 'Approach',
                      'line': {'color': 'black', 'width': 4}},
                     {'x': nan_line, 'y': nan_line, 'name': 'Takeoff',
                      'line': {'color': 'black', 'width': 4}},
                     {'x': nan_line, 'y': nan_line, 'name': 'Landing',
                      'line': {'color': 'grey', 'width': 4}},
                     {'x': nan_line, 'y': nan_line, 'name': 'Landing Transition',
                      'line': {'color': 'grey', 'width': 4}},
                     {'x': nan_line, 'y': nan_line, 'name': 'Flight',
                      'line': {'color': 'black', 'dash': 'dot'}},
                    ],
            'layout': layout}


@app.callback(Output('my-graph', 'figure'), inputs)
def update_graph(slope_angle, start_pos, approach_len, takeoff_angle,
                 fall_height):

    slope_angle = -float(slope_angle)
    start_pos = float(start_pos)
    approach_len = float(approach_len)
    takeoff_angle = float(takeoff_angle)
    fall_height = float(fall_height)

    try:
        surfs = make_jump(slope_angle, start_pos, approach_len, takeoff_angle,
                          fall_height)
    except InvalidJumpError as e:
        logging.error('Graph update error:', exc_info=e)
        return blank_graph

    slope, approach, takeoff, landing, trans, flight = surfs

    return {'data': [
                     {'x': slope.x, 'y': slope.y, 'name': 'Parent Slope',
                      'line': {'color': 'black', 'dash': 'dash'}},
                     {'x': approach.x, 'y': approach.y, 'name': 'Approach',
                      'line': {'color': 'black', 'width': 4}},
                     {'x': takeoff.x, 'y': takeoff.y, 'name': 'Takeoff',
                      'line': {'color': 'black', 'width': 4}},
                     {'x': landing.x, 'y': landing.y, 'name': 'Landing',
                      'line': {'color': 'grey', 'width': 4}},
                     {'x': trans.x, 'y': trans.y, 'name': 'Landing Transition',
                      'line': {'color': 'grey', 'width': 4}},
                     {'x': flight.x, 'y': flight.y, 'name': 'Flight',
                      'line': {'color': 'black', 'dash': 'dot'}},
                    ],
            'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
