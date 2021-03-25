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
from pkg_resources import packaging as pkg

import skijumpdesign
from skijumpdesign.functions import make_jump, cartesian_from_measurements
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

TITLE = ("Ski Jump Design and Analysis Tool "
         "for Specified Equivalent Fall Height")
VERSION_STAMP = 'skijumpdesign {}'.format(skijumpdesign.__version__)

ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'assets')
BOOTSTRAP_URL = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'
JQUERY_URL = 'https://code.jquery.com/jquery-3.5.1.min.js'
BOOTSTRAP_JS_URL = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js'

# NOTE : Turn the logger on to INFO level by default so it is recorded in any
# server logs.
logger = logging.getLogger('skijumpdesign')
logger.setLevel(logging.INFO)

# NOTE : ONHEROKU is a custom env variable that needs to be set via the app
# settings on heroku.com. This should be set as TRUE for the primary and
# staging apps.
if 'ONHEROKU' in os.environ:
    cmd_line_args = lambda x: None
    cmd_line_args.profile = False

    # NOTE : GATRACKINGID is a custom env variable that needs to be set via the
    # app settings on heroku.com. This should be set to a string corresponding
    # to the Google Analytics tracking id associated with the URL the app is
    # running on.
    if 'GATRACKINGID' in os.environ:
        ga_tracking_id = os.environ['GATRACKINGID']
        logging.info(ga_tracking_id)
        with open(os.path.join(ASSETS_PATH, 'gtag_template.js'), 'r') as f:
            ga_script_text = f.read()
        logging.info(ga_script_text)
        new_text = ga_script_text.format(ga_tracking_id=ga_tracking_id)
        logging.info(new_text)
        with open(os.path.join(ASSETS_PATH, 'gtag.js'), 'w') as f:
            f.write(new_text)
        GTAG_URL = '/assets/gtag.js'
        # TODO : Use dash's new assets folder capatility instead of all this
        # mess. The google code needs to be in the header.
        msg = 'Loaded google analytics script for {}.'.format(ga_tracking_id)
        logger.info(msg)
else:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument('-p', '--profile', action='store_true', default=False,
                        help='Profile the main callback with pyinstrument.')
    cmd_line_args = parser.parse_args()

    if cmd_line_args.profile:
        from pyinstrument import Profiler

# NOTE : Serve the file locally if it exists. Works for development and on
# heroku. It will not exist when installed via setuptools because the data file
# is placed at sys.prefix instead of into the site-packages directory. The
# backup is to serve from our git repo, but we must go through a third party to
# ensure that the content-type headers are correct, in this case:
# raw.githack.com. This may not be up-to-date due to caching. See
# https://gitlab.com/moorepants/skijumpdesign/issues/44 for more info.
if os.path.exists(os.path.join(ASSETS_PATH, 'skijump.css')):
    logging.info('Local css file found.')
    stylesheets = [BOOTSTRAP_URL]
else:
    logging.info('Local css file not found, loading from CDN.')
    URL_TEMP = ('https://glcdn.githack.com/moorepants/skijumpdesign/raw/'
                '{}/assets/skijump.css')
    if 'dev' in skijumpdesign.__version__:  # unlikely case
        CUS_URL = URL_TEMP.format('master')
    else:
        CUS_URL = URL_TEMP.format('v' + skijumpdesign.__version__)
    stylesheets = [BOOTSTRAP_URL, CUS_URL]

app = dash.Dash(__name__,
                suppress_callback_exceptions=True,  # needed for multipage
                external_stylesheets=stylesheets,
                external_scripts=[JQUERY_URL, BOOTSTRAP_JS_URL])
app.title = TITLE
server = app.server

###############################################################################
# INDEX LAYOUT
###############################################################################

nav_menu = \
    html.Nav([
        html.Div([
            html.Div([
                html.Button([
                    html.Span("Toggle Navigation", className="sr-only"),
                    html.Span("", className="icon-bar"),
                    html.Span("", className="icon-bar"),
                    html.Span("", className="icon-bar"),
                ], className="navbar-toggle collapsed",
                    **{"type": "button",
                       "data-toggle": "collapse",
                       "data-target": "#navbar",
                       "aria-expanded": "false",
                       "aria-controls": "navbar"}),
                html.Img(src=app.get_asset_url('logo-dark-background-50x50.png'),
                         className="navbar-brand"),
                html.A("Home", href="/",  className="navbar-brand"),
            ], className="navbar-header"),
            html.Div([
                html.Ul([
                    html.Li([dcc.Link('Ski Jump Design', href='/design')]),
                    html.Li([dcc.Link('Ski Jump Analysis', href='/analysis')]),
                ], className="nav navbar-nav"),
            ], id="navbar", className="collapse navbar-collapse"),
        ], className="container"),
    ], className="navbar navbar-inverse navbar-static-top",
        style={'background-color': 'rgb(64,71,86)',
               'border-color': 'rgb(64,71,86)'})

home_title = html.Div(
    [html.H1(TITLE,
             style={'text-align': 'center',
                    'padding-top': '20px',
                    'color': 'white'}),
     ],
    className='page-header',
    style={'height': 'auto',
           'margin-top': '-20px',
           'background': 'rgb(64, 71, 86)',
           })

ver_row = html.Div([html.P([html.Small(VERSION_STAMP)],
                           style={'text-align': 'right'})],
                   className='row')

home_button_design = html.A('Launch Design',
                            href='/design',
                            className='btn btn-primary btn-lg',
                            style={'padding': '44px 44px',
                                   'font-size': '36px'})

logo = html.Img(src=app.get_asset_url('logo-light-background-200x200.png'))

home_button_analysis = html.A('Launch Analysis',
                              href='/analysis',
                              className='btn btn-primary btn-lg',
                              style={'padding': '44px 44px',
                                     'font-size': '36px'})
home_buttons = html.Div([
    html.Div(home_button_design,
             style={'padding': '15px'},
             className='col-md-4 text-center'),
    html.Div(logo,
             className='col-md-4 text-center'),
    html.Div(home_button_analysis,
             style={'padding': '15px'},
             className='col-md-4 text-center'),
    ], className='row')

home_explanation_text = """\
# Explanation

This web application provides two tools to aid in the design and analysis of
ski jumps when one considers minimizing and controlling for the landing impact
speeds as defined by the "equivalent fall height" [1].

### Ski Jump Design

This tool allows the design of a ski jump that limits landing impact (measured
by a specified equivalent fall height[1]), for all takeoff speeds up to the
design speed. The calculated landing surface shape ensures that the jumper
always impacts the landing surface at the same perpendicular impact speed as if
dropped vertically from the specified equivalent fall height onto a horizontal
surface. This tool is described in [3].

### Ski Jump Analysis

Every jump landing surface shape has an associated equivalent fall height
function h(x) that characterizes the severity of impact at every possible
landing point with horizontal coordinate x. This tool allows calculation
of this function, once the shape of the landing surface and the takeoff
angle are specified, and thus allows the evaluation of the surface from
the point of view of impact severity.
"""

home_colophon_text = """\
# Colophon

This website was designed by Jason K. Moore, Mont Hubbard, and Bryn Cloud based
on theoretical and computational work detailed in [1]. A description of actual
fabrication of a constant equivlanet fall height jump is contained in [2].

The software that powers the website is open source and information on it can
be found here:

- [Download from PyPi.org](https://pypi.org/project/skijumpdesign)
- [Download from Anaconda.org](https://anaconda.org/conda-forge/skijumpdesign)
- [JOSS Journal Paper](https://doi.org/10.21105/joss.00818)
- Documentation: [skijumpdesign.readthedocs.io](http://skijumpdesign.readthedocs.io)
- Issue reports: [gitlab.com/moorepants/skijumpdesign/issues](https://gitlab.com/moorepants/skijumpdesign/issues)
- Source code repository: [gitlab.com/moorepants/skijumpdesign](http://gitlab.com/moorepants/skijumpdesign)

Contributions and issue reports are welcome!
"""

home_references_text = """\
# References

[1] Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. "A Design
Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall Height."
Sports Engineering 18, no. 4 (December 2015): 227–39.
[https://doi.org/10.1007/s12283-015-0182-6](https://doi.org/10.1007/s12283-015-0182-6)

[2] Petrone, N., Cognolato, M., McNeil, J.A., Hubbard, M. "Designing, building,
measuring and testing a constant equivalent fall height terrain park jump"
Sports Engineering 20, no. 4 (December 2017): 283-92.
[https://doi.org/10.1007/s12283-017-0253-y](https://doi.org/10.1007/s12283-017-0253-y)

[3] Moore, J. K. and Hubbard, M., (2018). skijumpdesign: A Ski Jump Design Tool
for Specified Equivalent Fall Height. Journal of Open Source Software, 3(28),
818, [https://doi.org/10.21105/joss.00818](https://doi.org/10.21105/joss.00818)

"""

home_feedback_text = """\
# Feedback

Bug reports, feature requests, and other general feedback can be submitted to
the [Gitlab issue tracker](https://gitlab.com/moorepants/skijumpdesign/issues)
or emailed directly to the authors at feedback@skijumpdesign.info.
"""

home_markdown = html.Div([
    html.Div([dcc.Markdown(home_explanation_text)], className='col-md-6'),
    html.Div([dcc.Markdown(home_colophon_text +
                           home_feedback_text)], className='col-md-6'),
],
                         className='row',
                         style={'background-color': 'rgb(64,71,86, 0.9)',
                                'color': 'white',
                                'padding-right': '20px',
                                'padding-left': '20px',
                                'margin-top': '40px',
                                'text-shadow': '1px 1px black',
                                })

colophon = html.Div([
    html.Div([dcc.Markdown(home_references_text)], className='col-md-12'),
    ],
    className='row',
    style={'background-color': 'rgb(64,71,86, 0.9)',
           'color': 'white',
           'padding-right': '20px',
           'padding-left': '20px',
           'margin-bottom': '40px',
           'text-shadow': '1px 1px black'})

layout_index = html.Div([nav_menu, home_title,
                         html.Div([ver_row, home_buttons, home_markdown,
                                   colophon],
                                  className='container')])

###############################################################################
# DESIGN LAYOUT
###############################################################################

if pkg.version.parse(dcc.__version__) < pkg.version.parse('1.0.0'):
    tooltip_kwarg = {}
else:
    tooltip_kwarg = {'tooltip': {'always_visible': True, 'placement': 'top'}}

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
        **tooltip_kwarg
        )
    ], style={'padding': '4px'})

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
        **tooltip_kwarg
        )
    ], style={'padding': '4px'})

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
        **tooltip_kwarg
        )
    ], style={'padding': '4px'})

takeoff_angle_widget = html.Div([
    html.H3('Takeoff Angle: 25 degrees',
            id='takeoff-text',
            style={'color': '#404756'}),
    dcc.Slider(
        id='takeoff_angle',
        min=-20,
        max=40,
        step=0.1,
        value=25,
        marks={-20: '-20 [deg]',
               -10: '-10 [deg]',
               0: '0 [deg]',
               10: '10 [deg]',
               20: '20 [deg]',
               30: '30 [deg]',
               40: '40 [deg]'},
        **tooltip_kwarg
        ),
    ], style={'padding': '4px'})

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

graph_widget = html.Div([
    dcc.Graph(id='my-graph',
              style={'width': '100%'},
              figure=go.Figure(layout=layout)),
    ], className='col-md-12')

row1 = html.Div([
                 html.H1(TITLE.replace(' and Analysis', ''),
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


row2 = html.Div([graph_widget], className='row')

build_dl_button = html.A('Download Profile for Building',
                         id='download-build-button',
                         href='',
                         className='btn btn-primary',
                         target='_blank',
                         download='',
                         )

analysis_dl_button = html.A('Download Profile for Analysis',
                            id='download-analysis-button',
                            href='',
                            className='btn btn-primary',
                            target='_blank',
                            download='',
                            )

row3 = html.Div([html.H2('Messages'), html.P('', id='message-text')],
                id='error-bar',
                className='alert alert-warning',
                style={'display': 'none'}
                )

loading_row = \
    html.Div([
        dcc.Loading([
            html.Div(children=[],
                     id='loading-area',
                     style={'height': '44px'},
                     className='col-md-12')
        ],
            id='test',
            type='dot',
            color='#c89b43',
            ),
    ], className='row')

loading_row_analysis = \
    html.Div([
        dcc.Loading([
            html.Div([],
                     id='loading-area-analysis',
                     style={'height': '44px'},
                     className='col-md-12')
        ],
            id='test',
            type='dot',
            color='#c89b43',
            ),
    ], className='row')

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
    html.Div([
        html.Div(build_dl_button, style={'padding': '10px'}),
        html.Div(analysis_dl_button, style={'padding': '10px'}),
    ], className='col-md-2'),
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
  shape provides the same impact EFH for all speeds up to and including this
  maximum achievable (design) speed.
- **Takeoff Angle**: The upward angle, relative to horizontal, at the end of
  the takeoff ramp, a free design parameter.
- **Equivalent Fall Height**: The desired equivalent fall height that
  characterizes landing impact everywhere on this jump.

## Outputs

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

The **Download Profile for Building** button returns a comma separated value
text file with two columns. In the downloaded file, the first column provides
the distance from the top of the jump (start of the takeoff curve) at every
meter along the slope and corresponding values of the height above the parent
slope in the second column. Both columns are in meters. This data is primarily
useful in building the actual jump, see [2].

The **Download Profile for Analysis** button returns a comma separated value
text file with two columns. In the downloaded file, the first column provides
the horizontal (x) distance from the takeoff point every 0.25 meters along the
horizontal and corresponding values of the vertical distance (y) from the
takeoff point in the second column. Both columns are in meters. This data can
be loaded directly into the analysis page.

For both downloads, the filename of the profile has the input parameters for
the jump; slope angle (sa), approach length (al), takeoff angle (ta), and
equivalent fall height (efh). These values are generated from the current
values of the sliders.

## Assumptions

The design calculations in this application depend on the ratios of aerodynamic
drag and snow friction resistive forces to inertial forces for the jumper, and
on estimates for reasonable turning accelerations (and their rates) able to be
borne by the jumper in the transitions (see reference [1] below). A list of
related assumed parameters with definitions and a set of nominal values for
these parameters is provided here:

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
- Choose the desired equivalent fall height (EFH), a measure of impact on
  landing (see reference [1] below). The landing surface shape
  calculated in the design provides the same EFH for all speeds up to and
  including the design speed and consequently for all starting points up to and
  including the maximum start position.
- Inspect and view the graph of the resulting jump design using the menu bar
  and iterate design parameters. The third button in the graph menu allows
  zoom.
- Download the jump design profile using one of the download buttons.

# References

[1] Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. "A Design
Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall Height."
Sports Engineering 18, no. 4 (December 2015): 227–39.
[https://doi.org/10.1007/s12283-015-0182-6](https://doi.org/10.1007/s12283-015-0182-6)

[2] Petrone, N., Cognolato, M., McNeil, J.A., Hubbard, M. "Designing, building,
measuring and testing a constant equivalent fall height terrain park jump"
Sports Engineering 20, no. 4 (December 2017): 283-92.
[https://doi.org/10.1007/s12283-017-0253-y](https://doi.org/10.1007/s12283-017-0253-y)

# Feedback

Bug reports, feature requests, and other general feedback can be submitted to
the [Gitlab issue tracker](https://gitlab.com/moorepants/skijumpdesign/issues)
or emailed directly to the authors at feedback@skijumpdesign.info>
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

layout_design = html.Div([nav_menu, row1,
                          html.Div([ver_row,
                                    row2,
                                    row3,
                                    loading_row,
                                    row4,
                                    row5,
                                    row6,
                                    row7,
                                    row8],
                                   className='container')])

###############################################################################
# ANALYSIS LAYOUT
###############################################################################

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
                       xaxis={'title': 'Distance [m]',
                              'zeroline': False,
                              # NOTE : Not sure why showgrid needs to be set
                              # explicitly here to have the grid lines display.
                              'showgrid': True},
                       yaxis={'scaleanchor': 'x',  # equal aspect ratio
                              'scaleratio': 1.0,  # equal aspect ratio
                              'title': 'Equivalent Fall Height [m]',
                              'zeroline': False},
                       legend={'orientation': "h",
                               'y': 1})

analysis_filename_widget = html.Div([
    html.H3('Filename:'),
    html.H4(id='filename-text-analysis'),
    html.H5(id='file-error',
            style={'color': 'red'})
])


def populated_efh_graph(takeoff_point, surface, distance, efh, speed):

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
         'text': ['Takeoff Speed: {:1.1f} m/s'.format(v) for v in speed],
         },
        {'x': distance,
         'y': distance_standards*recommend_efh,
         'name': 'Possible Soft Landing EFH',
         'line': {'color': '#404756', 'dash': 'dash'}},
        {'x': distance,
         'y': distance_standards * maximum_efh,
         'name': 'Knee Collapse EFH',
         'line': {'color': '#404756', 'dash': 'dot'}},
    ],
        'layout': layout_efh}


def blank_efh_graph(msg):
    nan_line = [np.nan]
    if layout.annotations != ():
        layout.annotations = ()
    data = {'data': [
                     {'x': [0.0, 0.0], 'y': [0.0, 0.0],
                      'name': 'Calculated EFH',
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


analysis_title_row = html.Div([
    html.H1(TITLE.replace('Design and', ''),
            style={'text-align': 'center',
                   'padding-top': '20px',
                   'color': 'white'}),
], className='page-header',
    style={
        'height': 'auto',
        'margin-top': '-20px',
        'background': 'rgb(64, 71, 86)',
    })

efh_graph_widget = html.Div(
    [dcc.Graph(id='efh-graph',
               style={'width': '100%'},
               figure=go.Figure(layout=layout_efh))],
    className='twelve columns')

table_widget = html.Div(id='datatable-upload')

analysis_takeoff_angle_widget = html.Div([
    html.H3('Takeoff Angle: [deg]',
            id='takeoff-text-analysis',
            style={'color': '#404756'},
            ),
    dcc.Input(id='takeoff_angle_analysis',
              placeholder='0',
              type='number',
              value='20',
              ),
])

compute_button = html.Div([
    html.Button('Run Analysis', id='compute-button',
                className='btn btn-primary btn-lg',),
    html.H5(id='compute-error', style={'color': 'red'}),
], style={'margin': '10px'})

download_efh_button = html.Div([
    html.A('Download EFH',
           id='download-efh-button',
           href='',
           className='btn btn-primary',
           target='_blank',
           download='efh_profile.csv')
], style={'margin': '10px'})

analysis_input_row = html.Div([
    html.Div([
        upload_widget,
        analysis_filename_widget,
        table_widget,
    ], className='col-md-8'),
    html.Div([
        analysis_takeoff_angle_widget,
        compute_button,
        download_efh_button,
    ], className='col-md-4 text-center'),
], className='row shaded')

analysis_graph_row = html.Div([efh_graph_widget], className='row')

markdown_text_analysis = """\
# Explanation

Every jump landing surface shape has an associated equivalent fall height
function h(x) that characterizes the severity of impact at every possible
landing point with horizontal coordinate x. This tool allows calculation
of this function, once the shape of the landing surface and the takeoff
angle are specified, and thus allows the evaluation of the surface from
the point of view of impact severity. A default jump is shown above that
happens to have a constant equivalent fall height for the primary landing
surface. Other jump designs can be uploaded and analyzed by following the
instructions below.

# Instructions

- Upload a csv or an Excel file containing the xy coordinates or the distance
  and angle measurements of the measured or proposed jump landing surface. The
  units of the surface coordinates must be meters or meters and degrees,
  respectively.
- Use the table to ensure the data file was uploaded properly.
- Set the angle of the takeoff ramp at the takeoff point and press the
  "Compute" button.
- Inspect and view the graph of the resulting jump profile and the calculated
  equivalent fall height. The third button in the graph menu allows zoom.

## Inputs

- **Upload**: A comma separated value (csv) file or an Excel spreadsheet file
  (.xls) in the format described above. The first row of the data file must be
  the column headers. No more than two columns should be present. The input
  data can be provided as one of two different types of surface measurements.
  This selection must match the type of measurement in the uploaded file.
  - **Cartesian (x,y)**: The x (horizontal) and y (vertical) coordinates of the
    jump cross-sectional profile. The coordinate pair (0.0, 0.0) is used as the
    takeoff point. The header of the first column should be `x` and the header
    of the second column should be `y`. The values in both columns should be
    expressed in the units of meters.
    ```
    x,y
    0.69,-0.29
    1.39,-0.56
    2.09,-0.80
    ```
    [Download an example file of this type](https://gitlab.com/moorepants/skijumpdesign/-/raw/master/docs/california-2002-surface.csv?inline=false).
  - **Distance & Angle**: The distance along the surface profile and the
    absolute angle at each distance measurement. The first (distance, angle)
    pair is used as the takeoff point. The header of the first column should be
    `distance` and the header of the second column should be `angle`. The
    distance values should be expressed in meters and the angle values should
    be expressed in degrees with positive values indicating an increasing slope
    and negative values a decreasing slope.
    ```
    distance,angle
    0.75,22.4
    0.25,14.6
    0.5,8.1
    ```
    [Download an example file of this type](https://gitlab.com/moorepants/skijumpdesign/-/raw/master/docs/sydney-measurements-2020.csv?inline=false)
- **Takeoff Angle**: The upward angle, relative to horizontal, at the end of
  the takeoff ramp.

### Table

The table allows inspection of the contents of the inputted csv or xls file
defining the landing surface shape.

### Graph

- **Jump Profile**: The jump profile displays the landing surface shape
  uploaded by the user.
- **Knee Collapse EFH**: This is the value of EFH (1.5 m) above which even
  elite ski jumpers are likely unable to prevent knee collapse. See reference
  [2] below.
- **Possible Soft Landing EFH**: This represents the 0.5 m recommended
  equivalent fall height for a possible soft landing EFH.
- **Calculated EFH**: This is the calculated equivalent fall height at 0.5 m
  horizontal intervals along the landing surface.

## Outputs

The output is a table of calculated EFH as a function of the horizontal
coordinate x of the landing point. This is plotted on the graph and can be
downloaded as a file named `efh_profile.csv` using the "Download EFH" button.

## Assumptions

The design calculations in this application depend on the ratios of aerodynamic
drag and snow friction resistive forces to inertial forces for the jumper, and
on estimates for reasonable turning accelerations (and their rates) able to be
borne by the jumper in the transitions (see reference [1] below). A list of
related assumed parameters with definitions and a set of nominal values for
these parameters is provided here:

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

# References

[1] Levy, Dean, Mont Hubbard, James A. McNeil, and Andrew Swedberg. "A Design
Rationale for Safer Terrain Park Jumps That Limit Equivalent Fall Height."
Sports Engineering 18, no. 4 (December 2015): 227–39.
[https://doi.org/10.1007/s12283-015-0182-6](https://doi.org/10.1007/s12283-015-0182-6)

[2] Minetti, A. E., L. P. Ardigò, D. Susta, and F. Cotelli. "Using Leg Muscles
as Shock Absorbers: Theoretical Predictions and Experimental Results of Drop
Landing Performance." Ergonomics 41, no. 12 (December 1, 1998): 1771–91.
https://doi.org/10.1080/001401398185965.

# Feedback

Bug reports, feature requests, and other general feedback can be submitted to
the [Gitlab issue tracker](https://gitlab.com/moorepants/skijumpdesign/issues)
or emailed directly to the authors at <feedback@skijumpdesign.info>.
"""

analysis_markdown_row = html.Div(
    [dcc.Markdown(markdown_text_analysis)],
    className='row',
    style={'background-color': 'rgb(64,71,86, 0.9)',
           'color': 'white', 'padding-right': '20px',
           'padding-left': '20px',
           'margin-top': '40px',
           'text-shadow': '1px 1px black',
           })

analysis_data_row = html.Div(id='output-data-upload',
                             style={'display': 'none'})

layout_analysis = html.Div([nav_menu, analysis_title_row,
                            html.Div([ver_row,
                                      analysis_graph_row,
                                      loading_row_analysis,
                                      analysis_input_row,
                                      analysis_markdown_row,
                                      analysis_data_row,
                                      ], className='container')
                            ])

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


def serve_layout():
    if flask.has_request_context():
        return url_bar_and_content_div
    # NOTE : This will cause an error "Duplicate component id found in the
    # initial layout" because this drops 3 id=navbar into the same layout for
    # newer versions of dash. This https://github.com/plotly/dash/pull/320 was
    # introduced around version 0.25.0 but this dual return method is needed
    # for 0.39.0, for example. My best guess is that when 1.0 hit maybe there
    # was some change that prevents this, so I use this function for less than
    # 1.0 and just serve the Location div for > 1.0. I didn't check versions
    # between 0.39.0 and 1.0 to see if this works.
    return html.Div([
        url_bar_and_content_div,
        layout_index,
        layout_design,
        layout_analysis,
    ])


if pkg.version.parse(dash.__version__) < pkg.version.parse('1.0'):
    app.layout = serve_layout
else:
    app.layout = url_bar_and_content_div

###############################################################################
# INDEX FUNCTIONALITY
###############################################################################


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/design":
        return layout_design
    elif pathname == "/analysis":
        return layout_analysis
    else:
        return layout_index

###############################################################################
# DESIGN FUNCTIONALITY
###############################################################################


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
    if layout.annotations != ():
        layout.annotations = ()
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

    # NOTE : analysis download, this should have the origin at the takeoff
    # point and give the coordinates of the entire jump surface (takeoff +
    # landing)
    built_surface = Surface(x, y)  # takeoff, landing, and transition
    built_surface.shift_coordinates(takeoff.x[0] - takeoff.x[-1],
                                    takeoff.y[0] - takeoff.y[-1])
    x_quarter_meter = np.arange(built_surface.start[0],
                                built_surface.end[0],
                                0.25)
    y_quarter_meter = built_surface.interp_y(x_quarter_meter)
    analysis_data = np.vstack((x_quarter_meter, y_quarter_meter)).T
    analysis_buf = BytesIO()
    np.savetxt(analysis_buf, analysis_data, fmt='%.2f', delimiter=',',
               newline="\n")
    analysis_header = ('x,y\n')
    analysis_file = analysis_header + analysis_buf.getvalue().decode()

    return header + buf.getvalue().decode(), analysis_file


@app.callback([Output('data-store', 'children'),
               Output('loading-area', 'children')],
              inputs)
def generate_data(slope_angle, approach_len, takeoff_angle, fall_height):

    if cmd_line_args.profile:
        profiler = Profiler()
        profiler.start()

    slope_angle = -float(slope_angle)
    approach_len = float(approach_len)
    takeoff_angle = float(takeoff_angle)
    fall_height = float(fall_height)

    blank_outputs = {'download': '#',
                     'analysis-download': '#',
                     'filename': 'build-profile.csv',
                     'analysis-filename': 'analysis-profile.csv',
                     'Takeoff Speed': 0.0,
                     'Snow Budget': 0.0,
                     'Flight Time': 0.0,
                     'Flight Distance': 0.0,
                     'Flight Height': 0.0}
    try:
        *surfs, outputs = make_jump(slope_angle, 0.0, approach_len,
                                    takeoff_angle, fall_height)
    except InvalidJumpError as e:
        logging.error('Graph update error:', exc_info=e)
        dic = blank_graph('<br>'.join(textwrap.wrap(str(e), 30)))
        dic['outputs'] = blank_outputs
    else:
        # NOTE : Move origin to start of takeoff.
        new_origin = surfs[2].start
        for surface in surfs:
            surface.shift_coordinates(-new_origin[0], -new_origin[1])
        dic = populated_graph(surfs)
        input_params = [-slope_angle, approach_len, takeoff_angle, fall_height]
        try:
            build_file, analysis_file = generate_csv_data(surfs)
        except InvalidJumpError as e:
            logging.error('Failed to create csv download data:', exc_info=e)
            dic = blank_graph('<br>'.join(textwrap.wrap(str(e), 30)))
            dic['outputs'] = blank_outputs
        else:
            outputs['download'] = build_file
            outputs['analysis-download'] = analysis_file
            fname = ("-profile-sa{:.1f}-al{:.1f}-ta{:.1f}-"
                     "efh{:.2f}.csv").format(*input_params)
            outputs['filename'] = "build" + fname
            outputs['analysis-filename'] = "analysis" + fname
            dic['outputs'] = outputs

    if cmd_line_args.profile:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))

    return json.dumps(dic, cls=PlotlyJSONEncoder), ''


@app.callback([Output('takeoff-speed-text', 'children'),
               Output('snow-budget-text', 'children'),
               Output('flight-time-text', 'children'),
               Output('flight-dist-text', 'children'),
               Output('flight-height-text', 'children'),
               Output('download-build-button', 'href'),
               Output('download-build-button', 'download'),
               Output('download-analysis-button', 'href'),
               Output('download-analysis-button', 'download'),
               Output('my-graph', 'figure')],
              [Input('data-store', 'children')])
def update_from_data_store(json_data):

    dic = json.loads(json_data)

    csv_string = dic['outputs']['download']
    csv_string = ("data:text/csv;charset=utf-8," +
                  urllib.parse.quote(csv_string))
    filename = dic['outputs']['filename']

    analysis_csv_string = dic['outputs']['analysis-download']
    analysis_csv_string = ("data:text/csv;charset=utf-8," +
                           urllib.parse.quote(analysis_csv_string))
    analysis_filename = dic['outputs']['analysis-filename']

    outputs = ['{:1.1f}'.format(dic['outputs']['Takeoff Speed']),
               '{:1.0f}'.format(dic['outputs']['Snow Budget']),
               '{:1.2f}'.format(dic['outputs']['Flight Time']),
               '{:1.1f}'.format(dic['outputs']['Flight Distance']),
               '{:1.1f}'.format(dic['outputs']['Flight Height']),
               csv_string,
               filename,
               analysis_csv_string,
               analysis_filename]

    del dic['outputs']

    outputs.append(dic)

    return outputs


###############################################################################
# ANALYSIS FUNCTIONALITY
###############################################################################


@app.callback(Output('filename-text-analysis', 'children'),
              [Input('upload-data', 'filename')])
def update_filename(filename):
    return '{}'.format(filename)


@app.callback(Output('file-error', 'children'),
              [Input('output-data-upload', 'children')])
def update_file_error(json_data):
    if json_data is None:
        return ''
    dic = json.loads(json_data)
    df = pd.read_json(dic, orient='index')

    if len(df.columns) > 2:
        return 'Only two columns can be present in the data file.'

    cols = tuple(sorted(df.columns))
    if cols == ('x', 'y'):
        pass
    elif cols == ('angle', 'distance'):
        pass
    else:
        return 'Column headers "{}" are incorrect'.format(cols)

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


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')])
def update_output(contents):
    if contents is not None:
        dic = parse_contents(contents)
        return dic


states_analysis = [
    State('output-data-upload', 'children'),
    State('takeoff_angle_analysis', 'value'),
]


@app.callback([Output('efh-graph', 'figure'),
               Output('compute-error', 'children'),
               Output('download-efh-button', 'href'),
               Output('loading-area-analysis', 'children')],
              [Input('compute-button', 'n_clicks'),
               Input('compute-button', 'children')],  # runs on load
              states_analysis)
def update_efh_graph(n_clicks, dummy, json_data, takeoff_angle):

    takeoff_angle = float(takeoff_angle)

    if json_data is None:  # no json_data on initial load
        # NOTE : Creates a default jump to plot, takeoff_angel of 10 degrees is
        # taken from default setting of input box.
        slope_angle, approach_len, fall_height = -15.0, 40.0, 0.8

        try:
            _, approach, takeoff, landing, landing_trans, _, _ = \
                make_jump(slope_angle, 0.0, approach_len, takeoff_angle,
                          fall_height)
        except InvalidJumpError:
            # NOTE : Should cause Surface to fail below.
            # TODO : Improve this, currently a poor workaround.
            x_vals = np.array([0.0, 1.0])
            y_vals = np.array([0.0, -1.0])
        else:
            delx = -(takeoff.end[0] - approach.start[0])
            dely = -(takeoff.end[1] - approach.start[1])
            landing.shift_coordinates(delx, dely)
            landing_trans.shift_coordinates(delx, dely)
            x_vals = np.hstack((landing.x, landing_trans.x[1:]))
            y_vals = np.hstack((landing.y, landing_trans.y[1:]))
    else:
        dic = json.loads(json_data)
        df = pd.read_json(dic, orient='index')
        if 'x' in df.columns:
            x_vals = df['x'].values
            y_vals = df['y'].values
        elif 'distance' in df.columns:
            logging.info('Converting distance and angle to x and y.')
            dist = df['distance'].values  # meters
            angles = df['angle'].values  # degrees
            x_vals, y_vals, _, _ = cartesian_from_measurements(
                dist, np.deg2rad(angles))
        else:
            raise ValueError('Incorrect columns in uploaded file.')

    takeoff_angle = np.deg2rad(takeoff_angle)
    takeoff_point = (0, 0)
    error_text = ''

    skier = Skier()

    try:
        surface = Surface(x_vals, y_vals)
        distance, efh, speed = surface.calculate_efh(takeoff_angle,
                                                     takeoff_point,
                                                     skier,
                                                     increment=0.5)
        update_graph = populated_efh_graph(takeoff_point, surface, distance,
                                           efh, speed)
        data = np.vstack((distance, efh)).T
    except Exception as e:
        update_graph = blank_efh_graph(e)
        data = np.vstack((np.nan, np.nan)).T
        error_text = 'There was an error processing this file: {}.'.format(e)

    # NOTE : StringIO() worked here for NumPy 1.14 but fails on NumPy 1.13,
    # thus BytesIO() is used as per an answer here:
    # https://stackoverflow.com/questions/22355026/numpy-savetxt-to-a-string
    buf = BytesIO()
    np.savetxt(buf, data, fmt='%.2f', delimiter=',', newline="\n")
    header = 'Distance Along Slope [m],EFH [m]\n'
    text = header + buf.getvalue().decode()
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(text)
    return update_graph, error_text, csv_string, ''


@app.callback(Output('datatable-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('output-data-upload', 'children')])
def update_table(contents, json_data):
    if contents is None:
        children_none = []
        return children_none
    else:
        dic = json.loads(json_data)
        df = pd.read_json(dic, orient='index')
        # dash-table 4.0 changed n_fixed_rows to fixed_rows
        datatable_kwargs = {
            'data': df.to_dict('rows'),
            'columns': [{'name': i, 'id': i} for i in df.columns],
            'style_table': {
                'maxHeight': '200',
                'overflowY': 'scroll',
            },
            'style_header': {'backgroundColor': 'rgba(96, 164, 255, 0.0)'},
        }
        if pkg.version.parse(dash_table.__version__) < pkg.version.parse('4.0'):
            datatable_kwargs['n_fixed_rows'] = 1
        else:
            datatable_kwargs['fixed_rows'] = {'headers': True, 'data': 0}
        children = [html.Div([dash_table.DataTable(**datatable_kwargs)])]
    return children


if __name__ == '__main__':
    if pkg.version.parse(dash.__version__) < pkg.version.parse('0.42.0'):
        app.run_server(debug=True)
    # NOTE : This turns off the feature that causes errors to be displayed in
    # the app instead of the terminal, giving pre 0.42.0 behavior.
    else:
        app.run_server(debug=True, dev_tools_ui=False)
