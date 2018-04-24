import os
import logging
import textwrap
import json
import urllib
from io import StringIO

import numpy as np
from scipy.interpolate import interp1d
import flask
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from skijumpdesign.functions import make_jump
from skijumpdesign.utils import InvalidJumpError

"""
Color Palette
https://mycolor.space/?hex=%2360A4FF&sub=1

#60a4ff rgb(96,164,255) : light blue
#404756 rgb(64,71,86) : dark blue grey
#a4abbd rgb(164,171,189) : light grey
#c89b43 : light yellow brown
#8e690a : brown

"""

STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BS_URL = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'
CUS_URL = 'https://moorepants.info/misc/skijump.css'
CUS_URL = '/static/skijump.css'

app = dash.Dash(__name__)
app.css.append_css({'external_url': [BS_URL, CUS_URL]})
server = app.server
if 'ONHEROKU' in os.environ:
    import dash_auth
    auth = dash_auth.BasicAuth(app, [['skiteam', 'howhigh']])


@app.server.route('/static/<resource>')
def serve_static(resource):
    _, ext = os.path.splitext(resource)
    if ext not in ['.css', '.js', '.png', 'svg']:
        return 'Invalid File Extension'
    else:
        return flask.send_from_directory(STATIC_PATH, resource)

approach_len_widget = html.Div([
    html.H3('Maximum Approach Length: 40 [m]',
            id='approach-len-text',
            style={'color': '#404756'}),
    dcc.Slider(
        id='approach_len',
        min=0,
        max=200,
        step=1,
        value=40,
        marks={0: '0 [m]',
               50: '50 [m]',
               100: '100 [m]',
               150: '150 [m]',
               200: '200 [m]'},
        )
    ])

fall_height_widget = html.Div([
    html.H3('Fall Height: 0.5 [m]',
            id='fall-height-text',
            style={'color': '#404756'}),
    dcc.Slider(
        id='fall_height',
        min=0.1,
        max=1.5,
        step=0.01,
        value=0.5,
        marks={0.10: '0.10 [m]',
               0.45: '0.45 [m]',
               0.80: '0.80 [m]',
               1.15: '1.15 [m]',
               1.5: '1.5 [m]'},
        )
    ])

slope_angle_widget = html.Div([
    html.H3('Parent Slope Angle: 15 degrees',
            id='slope-text',
            style={'color': '#404756'}),
    dcc.Slider(
        id='slope_angle',
        min=5,
        max=40,
        step=0.1,
        value=15,
        marks={5: '5 [deg]',
               12: '12 [deg]',
               19: '19 [deg]',
               25: '26 [deg]',
               32: '33 [deg]',
               40: '40 [deg]'},
        )
    ])

takeoff_angle_widget = html.Div([
    html.H3('Takeoff Angle: 25 degrees',
            id='takeoff-text',
            style={'color': '#404756'}),
    dcc.Slider(
        id='takeoff_angle',
        min=0,
        max=40,
        step=0.1,
        value=25,
        marks={0: '0 [deg]',
               10: '10 [deg]',
               20: '20 [deg]',
               30: '30 [deg]',
               40: '40 [deg]'},
        )
    ])

layout = go.Layout(autosize=False,
                   width=1000,
                   height=600,
                   hovermode='closest',
                   paper_bgcolor='rgba(96, 164, 255, 0.0)',  # transparent
                   plot_bgcolor='rgba(255, 255, 255, 0.5)',  # white
                   xaxis={'title': 'Distance [m]', 'zeroline': False},
                   yaxis={'scaleanchor': 'x',  # equal aspect ratio
                          'title': 'Height [m]', 'zeroline': False},
                   legend={'orientation': "h",
                           'y': 1.1})

graph_widget = html.Div([dcc.Graph(id='my-graph',
                                   figure=go.Figure(layout=layout))],
                        className='col-md-12')

row1 = html.Div([html.H1('Ski Jump Design Tool For Equivalent Fall Height',
                         style={'text-align': 'center',
                                'padding-top': '20px',
                                'color': 'white'})],
                className='page-header',
                style={
                       'height': '100px',
                       'margin-top': '-20px',
                       'background': 'rgb(64, 71, 86)',
                      })

row2 = html.Div([graph_widget], className='row')

button = html.A('Download Profile',
                id='download-button',
                href='',
                className='btn btn-primary',
                target='_blank',
                download='profile.csv')

row3 = html.Div([
    html.Div([], className='col-md-3'),
    html.Div([
        html.Table([
            html.Thead([
                html.Tr([html.Th('Output'),
                         html.Th('Value'),
                         html.Th('Unit')])]),
            html.Tbody([
                html.Tr([html.Td('Takeoff Speed'),
                         html.Td('', id='takeoff-speed-text'),
                         html.Td('m/s')]),
                html.Tr([html.Td('Flight Time'),
                         html.Td('', id='flight-time-text'),
                         html.Td('s')]),
                html.Tr([html.Td('Flight Distance'),
                         html.Td('', id='flight-dist-text'),
                         html.Td('m')]),
                html.Tr([html.Td('Flight Height'),
                         html.Td('', id='flight-height-text'),
                         html.Td('m')]),
                html.Tr([html.Td('Snow Budget'),
                         html.Td('', id='snow-budget-text'),
                         html.Td(['m', html.Sup('2')])])
            ]),
        ], className='table table-hover'),
    ], className='col-md-4'),
    html.Div([button], className='col-md-2'),
    html.Div([], className='col-md-3'),
], className='row')

row4 = html.Div([html.H2('Messages'), html.P('', id='message-text')],
                id='error-bar',
                className='alert alert-warning',
                style={'display': 'none'}
                )

row5 = html.Div([
                 html.Div([slope_angle_widget], className='col-md-5'),
                 html.Div([], className='col-md-2'),
                 html.Div([approach_len_widget], className='col-md-5'),
                 ], className='row', style={'margin-top': 15})

row6 = html.Div([
                 html.Div([takeoff_angle_widget], className='col-md-5'),
                 html.Div([], className='col-md-2'),
                 html.Div([fall_height_widget], className='col-md-5'),
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

> Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. "A Design
Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall Height."
Sports Engineering 18, no. 4 (December 2015): 227–39.
[https://doi.org/10.1007/s12283-015-0182-6](https://doi.org/10.1007/s12283-015-0182-6).

The software that powers the website is open source and information on it can
be found here:

- Documentation: [http://skijumpdesign.readthedocs.io]()
- Issue reports: [https://gitlab.com/moorepants/skijumpdesign/issues]()
- Source code repository: [http://gitlab.com/moorepants/skijumpdesign]()

Contributions and issue reports are welcome!

"""

row7 = html.Div([dcc.Markdown(markdown_text)],
                className='row',
                style={'background-color': 'rgb(64,71,86, 0.9)',
                       'color': 'white',
                       'padding-right': '20px',
                       'padding-left': '20px',
                       'margin-top': '40px',
                       'text-shadow': '1px 1px black',
                       })

row8 = html.Div(id='data-store', style={'display': 'none'})

app.layout = html.Div([row1,
                       html.Div([row2, row3, row4, row5, row6, row7, row8],
                                className='container')])


@app.callback(Output('slope-text', 'children'),
              [Input('slope_angle', 'value')])
def update_slope_text(slope_angle):
    slope_angle = float(slope_angle)
    return 'Parent Slope Angle: {:0.1f} [deg]'.format(slope_angle)


@app.callback(Output('approach-len-text', 'children'),
              [Input('approach_len', 'value')])
def update_approach_len_text(approach_len):
    approach_len = float(approach_len)
    return 'Maximum Approach Length: {:0.0f} [m]'.format(approach_len)


@app.callback(Output('takeoff-text', 'children'),
              [Input('takeoff_angle', 'value')])
def update_takeoff_text(takeoff_angle):
    takeoff_angle = float(takeoff_angle)
    return 'Takeoff Angle: {:0.1f} [deg]'.format(takeoff_angle)


@app.callback(Output('fall-height-text', 'children'),
              [Input('fall_height', 'value')])
def update_fall_height_text(fall_height):
    fall_height = float(fall_height)
    return 'Fall Height: {:0.2f} [m]'.format(fall_height)


inputs = [
          Input('slope_angle', 'value'),
          Input('approach_len', 'value'),
          Input('takeoff_angle', 'value'),
          Input('fall_height', 'value'),
         ]


def blank_graph(msg):
    nan_line = [np.nan]
    data = {'data': [
                     {'x': [0.0, 0.0], 'y': [0.0, 0.0], 'name': 'Parent Slope',
                      'text': ['Invalid Parameters<br>Error: {}'.format(msg)],
                      'mode': 'markers+text',
                      'textfont': {'size': 24},
                      'textposition': 'top',
                      'line': {'color': 'black', 'dash': 'dash'}},
                     {'x': nan_line, 'y': nan_line,
                      'name': 'Approach',
                      'line': {'color': '#404756', 'width': 4}},
                     {'x': nan_line, 'y': nan_line,
                      'name': 'Takeoff',
                      'line': {'color': '#a4abbd', 'width': 4}},
                     {'x': nan_line, 'y': nan_line,
                      'name': 'Landing',
                      'line': {'color': '#c89b43', 'width': 4}},
                     {'x': nan_line, 'y': nan_line,
                      'name': 'Landing Transition',
                      'line': {'color': '#8e690a', 'width': 4}},
                     {'x': nan_line, 'y': nan_line, 'name': 'Flight',
                      'line': {'color': 'black', 'dash': 'dot'}},
                    ],
            'layout': layout}
    return data


def populated_graph(surfs):

    slope, approach, takeoff, landing, trans, flight = surfs

    return {'data': [
                     {'x': slope.x.tolist(), 'y': slope.y.tolist(),
                      'name': 'Parent Slope',
                      'line': {'color': 'black', 'dash': 'dash'}},
                     {'x': approach.x.tolist(), 'y': approach.y.tolist(),
                      'name': 'Approach',
                      'line': {'color': '#a4abbd', 'width': 4}},
                     {'x': takeoff.x.tolist(), 'y': takeoff.y.tolist(),
                      'name': 'Takeoff',
                      'text': ['Height above parent: {:1.1f} m'.format(v) for v
                               in takeoff.height_above(slope)],
                      'line': {'color': '#8e690a', 'width': 4}},
                     {'x': landing.x.tolist(), 'y': landing.y.tolist(),
                      'name': 'Landing',
                      'text': ['Height above parent: {:1.1f} m'.format(v) for v
                               in landing.height_above(slope)],
                      'line': {'color': '#404756', 'width': 4}},
                     {'x': trans.x.tolist(), 'y': trans.y.tolist(),
                      'name': 'Landing Transition',
                      'text': ['Height above parent: {:1.1f} m'.format(v) for v
                               in trans.height_above(slope)],
                      'line': {'color': '#c89b43', 'width': 4}},
                     {'x': flight.pos[:, 0].tolist(),
                      'y': flight.pos[:, 1].tolist(),
                      'name': 'Flight',
                      'line': {'color': 'black', 'dash': 'dot'}},
                    ],
            'layout': layout}


def generate_csv_data(surfs):
    slope, approach, takeoff, landing, trans, flight = surfs
    x = np.hstack((takeoff.x, landing.x, trans.x))
    y = np.hstack((takeoff.y, landing.y, trans.y))
    f = interp1d(x, y, fill_value='extrapolate')
    x_one_meter = np.arange(takeoff.start[0], trans.end[0])
    h = f(x_one_meter) - slope.interp_y(x_one_meter)
    data = np.vstack((x_one_meter, h)).T
    s = StringIO()
    np.savetxt(s, data, fmt='%.2f', delimiter=',', newline="\n")
    return 'Horizontal Distance,Height Above Slope\n' + s.getvalue()


@app.callback(Output('data-store', 'children'), inputs)
def generate_data(slope_angle, approach_len, takeoff_angle, fall_height):
    slope_angle = -float(slope_angle)
    approach_len = float(approach_len)
    takeoff_angle = float(takeoff_angle)
    fall_height = float(fall_height)

    try:
        *surfs, outputs = make_jump(slope_angle, 0.0, approach_len,
                                    takeoff_angle, fall_height)
    except InvalidJumpError as e:
        logging.error('Graph update error:', exc_info=e)
        dic = blank_graph('<br>'.join(textwrap.wrap(str(e), 30)))
    else:
        # NOTE : Move origin to start of takeoff.
        new_origin = surfs[2].start
        for surface in surfs:
            surface.shift_coordinates(-new_origin[0], -new_origin[1])
        dic = populated_graph(surfs)

    outputs['download'] = generate_csv_data(surfs)

    dic['outputs'] = outputs
    return json.dumps(dic)


@app.callback(Output('my-graph', 'figure'), [Input('data-store', 'children')])
def update_graph(json_data):
    dic = json.loads(json_data)
    del dic['outputs']
    return dic


@app.callback(Output('takeoff-speed-text', 'children'),
              [Input('data-store', 'children')])
def update_takeoff_speed(json_data):
    dic = json.loads(json_data)
    return '{:1.1f}'.format(dic['outputs']['Takeoff Speed'])


@app.callback(Output('snow-budget-text', 'children'),
              [Input('data-store', 'children')])
def update_snow_budget(json_data):
    dic = json.loads(json_data)
    return '{:1.0f}'.format(dic['outputs']['Snow Budget'])


@app.callback(Output('flight-time-text', 'children'),
              [Input('data-store', 'children')])
def update_flight_time(json_data):
    dic = json.loads(json_data)
    return '{:1.2f}'.format(dic['outputs']['Flight Time'])


@app.callback(Output('flight-dist-text', 'children'),
              [Input('data-store', 'children')])
def update_flight_dist(json_data):
    dic = json.loads(json_data)
    return '{:1.1f}'.format(dic['outputs']['Flight Distance'])


@app.callback(Output('flight-height-text', 'children'),
              [Input('data-store', 'children')])
def update_flight_height(json_data):
    dic = json.loads(json_data)
    return '{:1.1f}'.format(dic['outputs']['Flight Height'])


@app.callback(Output('download-button', 'href'),
              [Input('data-store', 'children')])
def update_download_link(json_data):
    dic = json.loads(json_data)
    csv_string = dic['outputs']['download']
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string

if __name__ == '__main__':
    app.run_server(debug=True)
