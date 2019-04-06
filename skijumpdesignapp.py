import os
import logging
import textwrap
import json
import urllib
import argparse
from io import BytesIO, StringIO
from base64 import b64decode

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from xlrd import XLRDError
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import flask
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import skijumpdesign
from skijumpdesign.functions import make_jump
from skijumpdesign.surfaces import Surface
from skijumpdesign.skiers import Skier
from skijumpdesign.utils import InvalidJumpError

"""
Color Palette
https://mycolor.space/?hex=%2360A4FF&sub=1

This was setup to match the color blue of the sky in the background image.

#60a4ff rgb(96,164,255) : light blue
#404756 rgb(64,71,86) : dark blue grey
#a4abbd rgb(164,171,189) : light grey
#c89b43 : light yellow brown
#8e690a : brown

"""

TITLE = "Ski Jump Design Tool for Specified Equivalent Fall Height"
VERSION_STAMP = 'skijumpdesign {}'.format(skijumpdesign.__version__)

STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BS_URL = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'

# NOTE : Serve the file locally if it exists. Works for development and on
# heroku. It will not exist when installed via setuptools because the data file
# is placed at sys.prefix instead of into the site-packages directory. The
# backup is to serve from our git repo, but we must go through a third party to
# ensure that the content-type headers are correct, in this case:
# raw.githack.com. This may not be up-to-date due to caching. See
# https://gitlab.com/moorepants/skijumpdesign/issues/44 for more info.
if os.path.exists(os.path.join(STATIC_PATH, 'skijump.css')):
    logging.info('Local css file found.')
    CUS_URL = '/static/skijump.css'
else:
    logging.info('Local css file not found, loading from CDN.')
    URL_TEMP = ('https://glcdn.githack.com/moorepants/skijumpdesign/raw/'
                '{}/static/skijump.css')
    if 'dev' in skijumpdesign.__version__:  # unlikely case
        CUS_URL = URL_TEMP.format('master')
    else:
        CUS_URL = URL_TEMP.format('v' + skijumpdesign.__version__)

if 'ONHEROKU' in os.environ:
    cmd_line_args = lambda x: None
    cmd_line_args.profile = False
else:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument('-p', '--profile', action='store_true', default=False,
                        help='Profile the main callback with pyinstrument.')
    cmd_line_args = parser.parse_args()

    if cmd_line_args.profile:
        from pyinstrument import Profiler

app = dash.Dash(__name__)
app.css.append_css({'external_url': [BS_URL, CUS_URL]})
app.title = TITLE
server = app.server


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
    html.H3('Equivalent Fall Height: 0.5 [m]',
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
        ),
    ])

layout = go.Layout(autosize=True,
                   hovermode='closest',
                   paper_bgcolor='rgba(96, 164, 255, 0.0)',  # transparent
                   plot_bgcolor='rgba(255, 255, 255, 0.5)',  # white
                   xaxis={'title': 'Distance [m]', 'zeroline': False},
                   yaxis={'scaleanchor': 'x',  # equal aspect ratio
                          'scaleratio': 1.0,  # equal aspect ratio
                          'title': 'Height [m]', 'zeroline': False},
                   legend={'orientation': "h",
                           'y': 1.15})

graph_widget = html.Div([dcc.Graph(id='my-graph',
                                   # following is a trick to get height to
                                   # scale with width using padding-bottom
                                   style={'width': '100%',
                                          'height': '0',
                                          # NOTE : If less that 75% graphs may
                                          # not have any height on a phone.
                                          'padding-bottom': '75%'
                                          },
                                   figure=go.Figure(layout=layout))],
                        className='col-md-12')

row1 = html.Div([
                 html.H1(TITLE,
                         style={'text-align': 'center',
                                'padding-top': '20px',
                                'color': 'white'}),
                ],
                className='page-header',
                style={
                       'height': 'auto',
                       'margin-top': '-20px',
                       'background': 'rgb(64, 71, 86)',
                      })


row2 = html.Div([
                 graph_widget
                ], className='row')

button = html.A('Download Profile',
                id='download-button',
                href='',
                className='btn btn-primary',
                target='_blank',
                download='profile.csv')

row3 = html.Div([html.H2('Messages'), html.P('', id='message-text')],
                id='error-bar',
                className='alert alert-warning',
                style={'display': 'none'}
                )

row4 = html.Div([
                 html.Div([slope_angle_widget], className='col-md-5'),
                 html.Div([], className='col-md-2'),
                 html.Div([approach_len_widget], className='col-md-5'),
                 ], className='row shaded')

row5 = html.Div([
                 html.Div([takeoff_angle_widget], className='col-md-5'),
                 html.Div([], className='col-md-2'),
                 html.Div([fall_height_widget], className='col-md-5'),
                 ], className='row shaded')

row6 = html.Div([
    html.Div([], className='col-md-3'),
    html.Div([
        html.Table([
            html.Thead([
                html.Tr([html.Th('Outputs'),
                         html.Th('Value'),
                         html.Th('Unit')])]),
            html.Tbody([
                html.Tr([html.Td('Max Takeoff Speed'),
                         html.Td('', id='takeoff-speed-text'),
                         html.Td('m/s')]),
                html.Tr([html.Td('Max Flight Time'),
                         html.Td('', id='flight-time-text'),
                         html.Td('s')]),
                html.Tr([html.Td('Max Flight Distance'),
                         html.Td('', id='flight-dist-text'),
                         html.Td('m')]),
                html.Tr([html.Td('Max Flight Height Above Snow'),
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
], className='row shaded', style={'padding-top': '40px'})

markdown_text = """\
# Explanation

This tool allows the design of a ski jump that limits landing impact (measured
by a specified equivalent fall height[1]), for all takeoff speeds up to the
design speed. The calculated landing surface shape ensures that the jumper
always impacts the landing surface at the same perpendicular impact speed as if
dropped vertically from the specified equivalent fall height onto a horizontal
surface.

## Inputs

- **Parent Slope Angle**: The measured downward angle of the parent slope where
  the jump is desired. The designed jump shape is measured from this line.
- **Maximum Approach Length**: The maximum distance along the slope above the
  jump that the jumper can slide to build up speed. The jumper reaches a
  theoretical maximum speed at the end of this approach and the landing surface
  shape provides the same impact efh for all speeds up to and including this
  maximum achievable (design) speed.
- **Takeoff Angle**: The upward angle, relative to horizontal, at the end of
  the takeoff ramp, a free design parameter.
- **Equivalent Fall Height**: The desired equivalent fall height that
  characterizes landing impact everywhere on this jump.

## Outputs

*(all curves specified as x,y coordinates in a system with origin at the TO
point). All outputs are 2D curves. The complete jump shape consists of three;
the takeoff, landing and landing transition surfaces.*

### Graph

- **Takeoff Surface**: This transition curve is designed to give a smoothly
  varying acceleration transition from the parent slope to the takeoff point
  where the jumper begins flight.
- **Landing Surface**: This curve ensures that jumpers, launching at any speed
  from 0 m/s up to the maximum achievable (design) speed at the end of the
  approach, always impact the landing surface with a perpendicular speed no
  greater than the impact speed after falling from the equivalent vertical fall
  height onto a horizontal surface.
- **Landing Transition Surface**: This surface ensures a smooth and limited
  acceleration transition from  the landing surface back to the parent surface.
- **Flight Trajectory**: This is the jumper flight path corresponding to the
  design takeoff speed.

### Table

The table provides a set of outputs about the currently visible jump design:

- **Max Takeoff Speed**: This is the maximum speed the jumper can reach at the
  takeoff point when beginning from the top of the approach at a standstill.
  This speed dictates the maximum flight trajectory.
- **Max Flight Time**: The maximum time the jumper can be in the air given the
  maximum takeoff speed.
- **Max Flight Distance**: The maximum distance the jumper can jump given the
  maximum takeoff speed.
- **Max Flight Height Above Snow**: The maximum height the jumper can obtain
  above the landing surface snow given the maximum takeoff speed.
- **Snow Budget**: The cross sectional area of the snow under the takeoff and
  landing surfaces. Multiply this value times the width of the jump to obtain
  the volume of snow in the jump design.

### Profile

The **Download Profile** button returns a comma separated value text file with
two columns. The first column provides the distance from the top of the jump
(start of the takeoff curve) at every meter along the slope and corresponding
values of the height above the parent slope in the second column. Both columns
are in meters. This data is primarily useful in building the actual jump, see
[2].

## Assumptions

The design calculations in this application depend on the ratios of aerodynamic
drag and snow friction resistive forces to inertial forces for the jumper, and
on estimates for reasonable turning accelerations (and their rates) able to be
borne by the jumper in the transitions (see reference [1]). A list of related
assumed parameters with definitions and a set of nominal values for these
parameters is provided here:

- skier mass: 75.0 kg
- skier cross sectional area: 0.34 meters squared
- skier drag coefficient: 0.821
- snow/ski Coulomb friction coefficient: 0.03
- tolerable normal acceleration in approach-takeoff transition: 1.5 g's
- tolerable normal acceleration in landing transition: 3.0  g's
- fraction of the approach turning angle subtended by the circular section:
  0.99
- equilibration time the jumper should have on the straight ramp just before
  takeoff: 0.25 sec

# Instructions

- Select a parent slope angle to match or closely approximate the location
  where the jump is planned. The shape of the jump surface above this line is
  calculated.
- Set the length of approach to be the maximum distance along the parent slope
  from above the jump (measured from the top of the takeoff transition curve)
  that the jumper can descend when starting from rest. This distance determines
  the design (maximum) takeoff speed.
- Set the desired takeoff (TO) angle of the ramp at the takeoff point. This is
  a free design parameter but rarely are takeoff angles greater than 30 deg
  used.
- Choose the desired equivalent fall height (efh), a measure of impact on
  landing (see [1]). The landing surface shape calculated in the design
  provides the same efh for all speeds up to and including the design speed and
  consequently for all starting points up to and including the maximum start
  position.
- Inspect and view the graph of the resulting jump design using the menu bar
  and iterate design parameters. The third button allows zoom.
- Download the jump design profile using the **Download Profile** button.

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

ver_row = html.Div([html.P([html.Small(VERSION_STAMP)],
                           style={'text-align': 'right'})],
                   className='row')

# Analysis

upload_widget = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files', style={'color': 'blue'})
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    )
])

layout_efh = go.Layout(autosize=True,
                       hovermode='closest',
                       paper_bgcolor='rgba(96, 164, 255, 0.0)',  # transparent
                       plot_bgcolor='rgba(255, 255, 255, 0.5)',  # white
                       xaxis={'title': 'Distance [m]', 'zeroline': False},
                       yaxis={'scaleanchor': 'x',  # equal aspect ratio
                              'scaleratio': 1.0,  # equal aspect ratio
                              'title': 'EFH [m]', 'zeroline': False},
                       legend={'orientation': "h",
                               'y': 1})

analysis_filename_widget = html.Div([
    html.H3(id='filename-text-analysis'),
    html.H5(id='file-error',
            style={'color': 'red'})
])

analysis_takeoff_angle_widget = html.Div([
    html.H3('Takeoff Angle: [deg]',
            id='takeoff-text-analysis',
            style={'color': '#404756'}),
    dcc.Input(
        id='takeoff_angle_analysis',
        placeholder='0',
        type='number',
        value='0'
    ),
])

analysis_takeoff_x_widget = html.Div([
    html.H3('Takeoff Point, Distance: [m]',
            id='takeoff-text-distance',
            style={'color': '#404756'}),
    dcc.Input(
        id='takeoff_pos_dist',
        placeholder='0',
        type='number',
        value='0'
    ),
])

analysis_takeoff_y_widget = html.Div([
    html.H3('Takeoff Point, Height: [m]',
            id='takeoff-text-height',
            style={'color': '#404756'}),
    dcc.Input(
        id='takeoff_pos_height',
        placeholder='0',
        type='number',
        value='0'
    ),
])

def populated_efh_graph(takeoff_point, surface, distance, efh):

    recommend_efh = 0.5
    maximum_efh = 1.5
    distance_standards = np.ones(len(distance))

    layout_efh['annotations'] = [
        {
            'x': takeoff_point[0],
            'y': takeoff_point[1],
            'xref': 'x',
            'yref': 'y',
            'text': 'Takeoff Point',
        },
    ]

    return {'data': [
        {'x': surface.x,
         'y': surface.y,
         'name': 'Jump Profile',
         'line': {'color': '#8e690a', 'width': 4},
         'mode': 'lines'},
        {'x': distance,
         'y': efh,
         'name': 'Calculated EFH',
         'type': 'bar',
         'marker': {'color': '#c89b43'},
         },
        {'x': distance,
         'y': distance_standards*recommend_efh,
         'name': 'Recommended EFH',
         'line': {'color': '#404756', 'dash': 'dash'}},
        {'x': distance,
         'y': distance_standards * maximum_efh,
         'name': 'Maximum EFH',
         'line': {'color': '#404756', 'dash': 'dot'}},
    ],
        'layout': layout_efh}

def blank_efh_graph(msg):
    nan_line = [np.nan]
    if layout['annotations']:
        del layout['annotations']
    data = {'data': [
                     {'x': [0.0, 0.0], 'y': [0.0, 0.0], 'name': 'Calculated EFH',
                      'text': ['Invalid Parameters<br>Error: {}'.format(msg)],
                      'mode': 'markers+text',
                      'textfont': {'size': 24},
                      'textposition': 'top',
                      'line': {'color': '#c89b43'}},
                     {'x': nan_line, 'y': nan_line,
                      'name': 'Jump Profile',
                      'line': {'color': '#8e690a', 'width': 4}},
                     {'x': nan_line, 'y': nan_line,
                      'name': 'Recommended EFH',
                      'line': {'color': '#404756', 'dash': 'dash'}},
                     {'x': nan_line, 'y': nan_line,
                      'name': 'Maximum EFH',
                      'line': {'color': '#404756', 'dash': 'dot'}},
                    ],
            'layout': layout_efh}
    return data


def parse_contents(contents):
    content_type, content_string = contents.split(',')

    decoded = b64decode(content_string)

    try:
        df = pd.read_csv(
            StringIO(decoded.decode('utf-8')))
        dic = df.to_json(orient='index')
    except UnicodeDecodeError:
        try:
            df = pd.read_excel(BytesIO(decoded))
            dic = df.to_json(orient='index')
        except XLRDError as e:
            dic = blank_efh_graph('<br>'.join(textwrap.wrap(str(e), 30)))

    return json.dumps(dic, cls=PlotlyJSONEncoder)

efh_graph_widget = html.Div([dcc.Graph(id='efh-graph',
                                       style={'width': '100%',
                                              'height': '0',
                                              # NOTE : If less that 75% graphs may
                                              # not have any height on a phone.
                                              'padding-bottom': '75%'
                                              },
                                       figure=go.Figure(layout=layout_efh))],
                            className='twelve columns')

table_widget = html.Div(id='datatable-upload')

compute_button = html.Div([
    html.Button('Compute',
                id='compute-button',
                className='btn btn-primary',),
    html.H5(id='compute-error',
            style={'color': 'red'}),
    html.A('Download EFH',
           id='download-efh-button',
           href='',
           className='btn btn-primary',
           target='_blank',
           download='efh_profile.csv'),
])

analysis_title_row = html.Div([
    html.H1("Ski Jump Analysis",
            style={'text-align': 'center',
                   'padding-top': '20px',
                   'color': 'white'}),
],
    className='page-header',
    style={
        'height': 'auto',
        'margin-top': '-20px',
        'background': 'rgb(64, 71, 86)',
    })

analysis_upload_row = html.Div([
    upload_widget
], className='row')

analysis_takeoff_row = html.Div([
    html.Div([analysis_filename_widget], className='col-md-3'),
    html.Div([analysis_takeoff_angle_widget], className='col-md-3'),
    html.Div([analysis_takeoff_x_widget], className='col-md-3'),
    html.Div([analysis_takeoff_y_widget], className='col-md-3'),
], className='row shaded')

analysis_graph_row = html.Div([
    efh_graph_widget
], className='row')

analysis_table_row = html.Div([
    html.Div([table_widget], className='col-md-9'),
    html.Div([compute_button], className='col-md-3')
], className='row shaded')

markdown_text_analysis = """\
# Explanation

Every jump landing surface shape has an associated equivalent fall height
function h(x) that characterizes the severity of impact at every possible
landing point.  This tool allows the calculation of the function, once the
shape of the landing surface and the takeoff angle are specified, and thus
allows the evaluation of the surface from the point of impact severity.

## Inputs

- **Upload**: An excel or csv file of the x-y coordinates, relative to the
  horizontal, of the measured jump in meters. The first row of the data
  file must be the column headers. The first column must be the distance
  values of the jump along the horizontal and the second column must be the
  height values of the jump along the vertical.
- **Takeoff Angle**: The upward angle, relative to horizontal, at the end of
  the takeoff ramp.
- **Takeoff Point, Distance**: The distance, relative to the horizontal, of
  the takeoff point in meters.
- **Takeoff Point, Height**: The height, relative to the vertical, of the
  the takeoff point in meters.

## Outputs

*(all curves specified as x,y coordinates in a system with origin at the
takeoff point). All outputs are 2D curves. The complete jump profile
consists of the surfaces input by the user.*

### Graph

- **Jump Profile**: The jump profile displays the data uploaded by the user.
- **Maximum EFH**: This represents the maximum equivalent fall height a skier
  can feel without serious injury according to (Prof Hubbard enter here).
- **Recommended EFH**: This represents the 0.5 m recommended equivalent fall
  height recommended by (Prof Hubbard enter here).
- **Calculated EFH**: This is the calculated equivalent fall height at 0.2 m
  intervals after the user specified takeoff point.

### Table

The table provides a look at the inputted csv or excel file that is used for
efh calculations.

## Assumptions

The design calculations in this application depend on the ratios of aerodynamic
drag and snow friction resistive forces to inertial forces for the jumper, and
on estimates for reasonable turning accelerations (and their rates) able to be
borne by the jumper in the transitions (see reference [1]). A list of related
assumed parameters with definitions and a set of nominal values for these
parameters is provided here:

- skier mass: 75.0 kg
- skier cross sectional area: 0.34 meters squared
- skier drag coefficient: 0.821
- snow/ski Coulomb friction coefficient: 0.03
- tolerable normal acceleration in approach-takeoff transition: 1.5 g's
- tolerable normal acceleration in landing transition: 3.0  g's
- fraction of the approach turning angle subtended by the circular section:
  0.99
- equilibration time the jumper should have on the straight ramp just before
  takeoff: 0.25 sec

# Instructions

- Upload an excel or csv file of the x-y coordinates of the measured jump. The
  values must be in meters. The first row of the data file must have be the column
  headers. The first column must be the distance values of the jump along the
  horizontal  and the second column must be the height values of the jump along
  the vertical.
- Set the takeoff angle of the ramp at the takeoff point.
- Set the coordinates of the takeoff point relative to the uploaded data file.
- Inspect and view the graph of the resulting jump profile and the calculated
  equivalent fall height. The third button allows zoom.
- Use the table to ensure the data file was uploaded properly.

"""

analysis_markdown_row = html.Div([dcc.Markdown(markdown_text_analysis)],
                                 className='row',
                                 style={'background-color': 'rgb(64,71,86, 0.9)',
                                        'color': 'white', 'padding-right': '20px',
                                        'padding-left': '20px',
                                        'margin-top': '40px',
                                        'text-shadow': '1px 1px black',
                                        })

analysis_data_row = html.Div(id='output-data-upload', style={'display': 'none'})

# Home

home_title = html.Div([
    html.H1("Ski Jump Tool",
            style={'text-align': 'center',
                   'padding-top': '20px',
                   'color': 'white'}),
],
    className='page-header',
    style={
        'height': 'auto',
        'margin-top': '-20px',
        'background': 'rgb(64, 71, 86)',
    })

markdown_text_home = """\
# Explanation

### Ski Jump Design
This tool allows the design of a ski jump that limits landing impact (measured
by a specified equivalent fall height[1]), for all takeoff speeds up to the
design speed. The calculated landing surface shape ensures that the jumper
always impacts the landing surface at the same perpendicular impact speed as if
dropped vertically from the specified equivalent fall height onto a horizontal
surface.

### Ski Jump Analysis
Every jump landing surface shape has an associated equivalent fall height
function h(x) that characterizes the severity of impact at every possible
landing point.  This tool allows the calculation of the function, once the
shape of the landing surface and the takeoff angle are specified, and thus
allows the evaluation of the surface from the point of impact severity.

# Colophon

This website was designed by Jason K. Moore and Mont Hubbard based on
theoretical and computational work detailed in [1]. A description of actual
fabrication of such a jump is contained in [2].

The software that powers the website is open source and information on it can
be found here:

- [Download from PyPi.org](https://pypi.org/project/skijumpdesign)
- [Download from Anaconda.org](https://anaconda.org/conda-forge/skijumpdesign)
- Documentation: [http://skijumpdesign.readthedocs.io](http://skijumpdesign.readthedocs.io)
- Issue reports: [https://gitlab.com/moorepants/skijumpdesign/issues](https://gitlab.com/moorepants/skijumpdesign/issues)
- Source code repository: [http://gitlab.com/moorepants/skijumpdesign](http://gitlab.com/moorepants/skijumpdesign)

Contributions and issue reports are welcome!

# References

[1] Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. "A Design
Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall Height."
Sports Engineering 18, no. 4 (December 2015): 227–39.
[https://doi.org/10.1007/s12283-015-0182-6](https://doi.org/10.1007/s12283-015-0182-6)

[2] Petrone, N., Cognolato, M., McNeil, J.A., Hubbard, M. “Designing, building,
measuring and testing a constant equivalent fall height terrain park jump"
Sports Engineering 20, no. 4 (December 2017): 283-92.
[https://doi.org/10.1007/s12283-017-0253-y](https://doi.org/10.1007/s12283-017-0253-y)

"""

home_markdown = html.Div([dcc.Markdown(markdown_text_home)],

                         className='row',
                         style={'background-color': 'rgb(64,71,86, 0.9)',
                                'color': 'white',
                                'padding-right': '20px',
                                'padding-left': '20px',
                                'margin-top': '40px',
                                'text-shadow': '1px 1px black',
                                })

home_button_design = html.A('Ski Jump Design',
                            href='/design',
                            className='btn btn-primary btn-lg', style={'padding': '72px 72px'})

home_button_analysis = html.A('Ski Jump Analysis',
                              href='/analysis',
                              className='btn btn-primary btn-lg', style={'padding': '72px 72px'})

home_buttons = html.Div([
    html.Div([home_button_design], style={'display': 'inline-block', 'padding': '15px'}),
    html.Div([home_button_analysis], style={'display': 'inline-block', 'padding': '15px'}),
], className='row shaded', style={'padding': '40px', 'display': 'flex', 'justify-content': 'center'})

nav_menu = html.Div([
    html.Ul([
            html.Li([
                    dcc.Link('Home', href='/')
                    ], className='active'),
            html.Li([
                    dcc.Link('Ski Jump Design', href='/design')
                    ]),
            html.Li([
                    dcc.Link('Ski Jump Analysis', href='/analysis')
                    ]),
            ], className='nav navbar-nav')
], className='navbar navbar-expand-sm navbar-static-top', style={'background-color': 'rgb(64,71,86)'})

# Page Layouts

layout_index = html.Div([nav_menu, home_title,
                         html.Div([
                             ver_row, home_buttons, home_markdown
                         ], className='container')
                        ])

layout_design = html.Div([nav_menu, row1,
                          html.Div([ver_row, row2, row3, row4, row5, row6,
                                    row7, row8],
                                   className='container')])

layout_analysis = html.Div([nav_menu, analysis_title_row,
                            html.Div([ver_row,
                                      analysis_upload_row,
                                      analysis_takeoff_row,
                                      analysis_table_row,
                                      analysis_graph_row,
                                      analysis_markdown_row,
                                      analysis_data_row
                                      ], className='container')
                            ])

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

def serve_layout():
    if flask.has_request_context():
        return url_bar_and_content_div
    return html.Div([
        url_bar_and_content_div,
        layout_index,
        layout_design,
        layout_analysis,
    ])

app.layout = serve_layout

# Ski Jump Design Callbacks

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
    return 'Equivalent Fall Height: {:0.2f} [m]'.format(fall_height)


inputs = [
          Input('slope_angle', 'value'),
          Input('approach_len', 'value'),
          Input('takeoff_angle', 'value'),
          Input('fall_height', 'value'),
         ]


def blank_graph(msg):
    nan_line = [np.nan]
    if layout['annotations']:
        del layout['annotations']
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


def create_arc(x_cen, y_cen, radius, angle):
    """Returns the x and y coordinates of an arc that starts at the angled
    slope and ends at horizontal."""
    x_start = x_cen + radius * np.cos(angle)
    x_end = x_cen + radius
    x = np.linspace(x_start, x_end)
    y = -np.sqrt(radius**2 - (x - x_cen)**2) + y_cen
    return x, y


def populated_graph(surfs):

    slope, approach, takeoff, landing, trans, flight = surfs

    leader_len = (approach.x[-1] - approach.x[0]) / 3

    arc_x, arc_y = create_arc(*approach.start, 2 * leader_len / 3, slope.angle)

    layout['annotations'] = [
        {
         'x': takeoff.end[0],
         'y': takeoff.end[1],
         'xref': 'x',
         'yref': 'y',
         'text': 'Takeoff Point',
        },
        {
         'x': arc_x[35],
         'y': arc_y[35],
         'xref': 'x',
         'yref': 'y',
         'text': 'Parent Slope Angle',
         'ax': 80,
         'ay': 0,
        },
    ]

    return {'data': [
                     {'x': [approach.x[0], approach.x[0] + leader_len],
                      'y': [approach.y[0], approach.y[0]],
                      'line': {'color': 'black', 'width': 1},
                      'mode': 'lines',
                      'hoverinfo': 'none',
                      'showlegend': False},
                     {'x': arc_x.tolist(),
                      'y': arc_y.tolist(),
                      'line': {'color': 'black'},
                      'mode': 'lines',
                      'hoverinfo': 'none',
                      'showlegend': False},
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
                      'shape': 'spline',
                      'line': {'color': '#8e690a', 'width': 4}},
                     {'x': landing.x.tolist(), 'y': landing.y.tolist(),
                      'name': 'Landing',
                      'text': ['Height above parent: {:1.1f} m'.format(v) for v
                               in landing.height_above(slope)],
                      'line': {'color': '#404756', 'width': 4},
                      'shape': 'spline',
                      },
                     {'x': trans.x.tolist(), 'y': trans.y.tolist(),
                      'name': 'Landing Transition',
                      'text': ['Height above parent: {:1.1f} m'.format(v) for v
                               in trans.height_above(slope)],
                      'shape': 'spline',
                      'line': {'color': '#c89b43', 'width': 4}},
                     {'x': flight.pos[:, 0].tolist(),
                      'y': flight.pos[:, 1].tolist(),
                      'shape': 'spline',
                      'name': 'Flight',
                      'line': {'color': 'black', 'dash': 'dot'}},
                    ],
            'layout': layout}


def generate_csv_data(surfs):
    """Returns a csv string containing the height above the parent slope of the
    jump at one meter intervals along the slope from the top of the jump."""
    slope, approach, takeoff, landing, trans, flight = surfs

    x = np.hstack((takeoff.x, landing.x, trans.x))
    y = np.hstack((takeoff.y, landing.y, trans.y))

    f = interp1d(x, y, fill_value='extrapolate')

    # One meter intervals along the slope.
    hyp_one_meter = np.arange(0.0, (trans.end[0] - takeoff.start[0]) /
                              np.cos(slope.angle))
    # Corresponding x values for the one meter intervals along slope
    x_one_meter = takeoff.start[0] + hyp_one_meter * np.cos(slope.angle)

    height = f(x_one_meter) - slope.interp_y(x_one_meter)

    data = np.vstack((hyp_one_meter, height)).T
    # NOTE : StringIO() worked here for NumPy 1.14 but fails on NumPy 1.13,
    # thus BytesIO() is used as per an answer here:
    # https://stackoverflow.com/questions/22355026/numpy-savetxt-to-a-string
    buf = BytesIO()
    np.savetxt(buf, data, fmt='%.2f', delimiter=',', newline="\n")
    header = 'Distance Along Slope [m],Height Above Slope [m]\n'
    return header + buf.getvalue().decode()


@app.callback(Output('data-store', 'children'), inputs)
def generate_data(slope_angle, approach_len, takeoff_angle, fall_height):

    if cmd_line_args.profile:
        profiler = Profiler()
        profiler.start()

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
        dic['outputs'] = {'download': '#',
                          'Takeoff Speed': 0.0,
                          'Snow Budget': 0.0,
                          'Flight Time': 0.0,
                          'Flight Distance': 0.0,
                          'Flight Height': 0.0}
    else:
        # NOTE : Move origin to start of takeoff.
        new_origin = surfs[2].start
        for surface in surfs:
            surface.shift_coordinates(-new_origin[0], -new_origin[1])
        dic = populated_graph(surfs)
        outputs['download'] = generate_csv_data(surfs)
        dic['outputs'] = outputs

    if cmd_line_args.profile:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))

    return json.dumps(dic, cls=PlotlyJSONEncoder)


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

# Index Callbacks

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/design":
        return layout_design
    elif pathname == "/analysis":
        return layout_analysis
    else:
        return layout_index

# Analysis Callbacks

@app.callback(Output('filename-text-analysis', 'children'),
              [Input('upload-data', 'filename')])
def update_filename(filename):
    return 'Filename: {}'.format(filename)

@app.callback(Output('file-error', 'children'),
              [Input('output-data-upload', 'children')])
def update_file_error(json_data):
    if json_data is None:
        return ''
    dic = json.loads(json_data)
    df = pd.read_json(dic, orient='index')
    if df.isnull().sum().sum() > 0:
        return 'File has missing values.'
    elif type(df.columns[0]) != str or type(df.columns[1]) != str:
        return 'Make sure file has a row header.'
    else:
        return ''

@app.callback(Output('takeoff-text-analysis', 'children'),
              [Input('takeoff_angle_analysis', 'value')])
def update_takeoff_angle(takeoff_angle):
    takeoff_angle = float(takeoff_angle)
    return 'Takeoff Angle: {:0.1f} [deg]'.format(takeoff_angle), ''

@app.callback(Output('takeoff-text-distance', 'children'),
              [Input('takeoff_pos_dist', 'value')])
def update_takeoff_xpos(takeoff_pos_x):
    takeoff_pos_x = float(takeoff_pos_x)
    return 'Takeoff Point, Distance: {:0.1f} [m]'.format(takeoff_pos_x), ''

@app.callback(Output('takeoff-text-height', 'children'),
              [Input('takeoff_pos_height', 'value')])
def update_takeoff_ypos(takeoff_pos_y):
    takeoff_y = float(takeoff_pos_y)
    return 'Takeoff Point, Height: {:0.1f} [m]'.format(takeoff_y)

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')])
def update_output(contents):
    if contents is not None:
        dic = parse_contents(contents)
        return dic


states_analysis = [
    State('output-data-upload', 'children'),
    State('takeoff_angle_analysis', 'value'),
    State('takeoff_pos_dist', 'value'),
    State('takeoff_pos_height', 'value')
]


@app.callback([Output('efh-graph', 'figure'),
               Output('compute-error', 'children'),
               Output('download-efh-button', 'href'),],
              [Input('compute-button', 'n_clicks')],
              states_analysis)
def update_efh_graph(n_clicks, json_data, takeoff_angle, takeoff_point_x, takeoff_point_y):
    dic = json.loads(json_data)
    df = pd.read_json(dic, orient='index')

    surface = Surface(df.iloc[:, 0].values, df.iloc[:, 1].values)
    skier = Skier()
    takeoff_angle = float(takeoff_angle)
    takeoff_angle = np.deg2rad(takeoff_angle)
    takeoff_point_x = float(takeoff_point_x)
    takeoff_point_y = float(takeoff_point_y)
    takeoff_point = (takeoff_point_x, takeoff_point_y)

    try:
        distance, efh = surface.calculate_efh(takeoff_angle, takeoff_point, skier)
        update_graph = populated_efh_graph(takeoff_point, surface, distance, efh)
        data = np.vstack((distance, efh)).T
        error_text = ''
    except Exception as e:
        update_graph = blank_efh_graph(e)
        data = np.vstack((np.nan, np.nan)).T
        error_text = 'There was an error processing this file.'

    # NOTE : StringIO() worked here for NumPy 1.14 but fails on NumPy 1.13,
    # thus BytesIO() is used as per an answer here:
    # https://stackoverflow.com/questions/22355026/numpy-savetxt-to-a-string
    buf = BytesIO()
    np.savetxt(buf, data, fmt='%.2f', delimiter=',', newline="\n")
    header = 'Distance Along Slope [m],EFH [m]\n'
    text = header + buf.getvalue().decode()
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(text)
    return update_graph, error_text, csv_string


@app.callback(Output('datatable-upload', 'children'),
              [Input('upload-data', 'contents'), Input('output-data-upload', 'children')])
def update_table(contents, json_data):
    if contents is None:
        children_none = []
        return children_none
    else:
        dic = json.loads(json_data)
        df = pd.read_json(dic, orient='index')
        children = [
            html.Div([
                dash_table.DataTable(
                    data=df.to_dict('rows'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    n_fixed_rows=1,
                    style_table={
                        'maxHeight': '200',
                        'overflowY': 'scroll',
                    },
                    style_header={'backgroundColor': 'rgba(96, 164, 255, 0.0)'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }]
                ),
            ])
        ]
    return children

if __name__ == '__main__':
    app.run_server(debug=True)
