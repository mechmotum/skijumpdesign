import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from safeskijump.functions import make_jump

BS_URL = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'

app = dash.Dash()
app.css.append_css({'external_url': BS_URL})

start_pos_widget = html.Div([
    html.P('Start Position [m]'),
    dcc.Input(id='start_pos',
              placeholder='Start Position [meters]',
              type='number',
              value='10')
    ])

approach_len_widget = html.Div([
    html.P('Approach Length [m]'),
    dcc.Input(id='approach_len',
              placeholder='Approach Length [meters]',
              type='number',
              value='50')
    ])

fall_height_widget = html.Div([
    html.P('Fall Height [m]'),
    dcc.Input(id='fall_height',
              placeholder='Fall Height [meters]',
              type='number',
              value='0.2')
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

controls_widget = html.Div([start_pos_widget, approach_len_widget,
                            fall_height_widget, slope_angle_widget,
                            takeoff_angle_widget],
                           className='col-md-4')

layout = go.Layout(autosize=False,
                   width=800,
                   height=800,
                   yaxis={'scaleanchor': 'x'})  # equal aspect ratio

graph_widget = html.Div([dcc.Graph(id='my-graph',
                                   figure=go.Figure(layout=layout))],
                        className='col-md-4')

app.layout = html.Div([html.H1('Ski Jump Design'),
                       html.Div([controls_widget, graph_widget],
                                className='row')],
                      className='container')

inputs = [Input('slope_angle', 'value'),
          Input('start_pos', 'value'),
          Input('approach_len', 'value'),
          Input('takeoff_angle', 'value'),
          Input('fall_height', 'value')]


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


@app.callback(Output('my-graph', 'figure'), inputs)
def update_graph(slope_angle, start_pos, approach_len, takeoff_angle,
                 fall_height):

    slope_angle = float(slope_angle)
    start_pos = float(start_pos)
    approach_len = float(approach_len)
    takeoff_angle = float(takeoff_angle)
    fall_height = float(fall_height)

    surfs = make_jump(-slope_angle, start_pos, approach_len, takeoff_angle,
                      fall_height)
    slope, approach, takeoff, landing, landtrans, flight = surfs

    return {'data': [
                     {'x': slope.x, 'y': slope.y, 'name': 'Slope'},
                     {'x': approach.x, 'y': approach.y, 'name': 'Approach'},
                     {'x': takeoff.x, 'y': takeoff.y, 'name': 'Takeoff'},
                     {'x': landing.x, 'y': landing.y, 'name': 'Landing'},
                     {'x': landtrans.x, 'y': landtrans.y, 'name': 'Landing Transition'},
                     {'x': flight.x, 'y': flight.y, 'name': 'Flight'},
                    ],
            'layout': layout}

if __name__ == '__main__':
    app.run_server()
