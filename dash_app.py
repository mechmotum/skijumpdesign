import numpy as np
import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from safeskijump.functions import *

layout = go.Layout(autosize=False, width=800, height=800, yaxis={'scaleanchor': 'x'})
fig = go.Figure(layout=layout)

app = dash.Dash()
app.css.append_css({'external_url': 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css'})

app.layout = \
    html.Div([
        html.Div([
            html.Div([
                html.H1('Ski Jump Design'),
                html.P('Start Position [m]'),
                dcc.Input(
                    id='start_pos',
                    placeholder='Start Position [meters]',
                    type='number',
                    value='10'
                ),
                html.P('Approach Length [m]'),
                dcc.Input(
                    id='approach_len',
                    placeholder='Approach Length [meters]',
                    type='number',
                    value='50'
                ),
                html.P('Slope Angle: 10 degrees', id='slope-text'),
                dcc.Slider(
                    id='slope_angle',
                    min=0,
                    max=45,
                    step=1,
                    value=10,
                    marks={0: '0 [deg]', 45: '45 [deg]'},
                    updatemode='drag'
                ),
                html.P('Takeoff Angle: 10 degrees', id='takeoff-text'),
                dcc.Slider(
                    id='takeoff_angle',
                    min=0,
                    max=45,
                    step=1,
                    value=20,
                    marks={0: '0 [deg]', 45: '45 [deg]'},
                    updatemode='drag'
                ),
                ],
                className='col-md-4'),
            html.Div([
                dcc.Graph(id='my-graph', figure=fig)],
                className='col-md-4'),
        ], className='row'),
    ], className='container')


def make_jump(slope_angle, start_pos, approach_len, takeoff_angle):

    tolerable_acc = 1.5  # G

    takeoff_entry_speed = compute_approach_exit_speed(
        slope_angle, start_pos, approach_len)

    curve_x, curve_y, _, _ = generate_takeoff_curve(
        slope_angle, takeoff_entry_speed, takeoff_angle, tolerable_acc)

    ramp_entry_speed = compute_design_speed(
        takeoff_entry_speed, slope_angle, curve_x, curve_y)

    curve_x, curve_y = add_takeoff_ramp(
        takeoff_angle, ramp_entry_speed, curve_x, curve_y)

    design_speed = compute_design_speed(
        takeoff_entry_speed, slope_angle, curve_x, curve_y)

    init_x = (start_pos + approach_len) * np.cos(np.deg2rad(slope_angle)) + curve_x[-1]
    init_y = -(start_pos + approach_len) * np.sin(np.deg2rad(slope_angle)) + curve_y[-1]

    traj_x, traj_y, vel_x, vel_y = compute_flight_trajectory(
        slope_angle, (init_x, init_y), takeoff_angle, design_speed)

    return curve_x, curve_y, traj_x, traj_y


def create_plot_arrays(slope_angle, start_pos, approach_len, takeoff_angle,
                       takeoff_curve_x, takeoff_curve_y, flight_traj_x,
                       flight_traj_y):

    # plot approach
    l = np.linspace(start_pos, start_pos + approach_len)
    x = l * np.cos(np.deg2rad(slope_angle))
    y = -l * np.sin(np.deg2rad(slope_angle))
    approach_xy = (x, y)

    # plot takeoff curve
    shifted_takeoff_curve_x = takeoff_curve_x + x[-1]
    shifted_takeoff_curve_y = takeoff_curve_y + y[-1]
    takeoff_xy = (shifted_takeoff_curve_x, shifted_takeoff_curve_y)

    # plot takeoff angle line
    takeoff_line_slope = np.tan(np.deg2rad(takeoff_angle))
    takeoff_line_intercept = (shifted_takeoff_curve_y[-1] - takeoff_line_slope *
                              shifted_takeoff_curve_x[-1])

    x_takeoff = np.linspace(shifted_takeoff_curve_x[0],
                            shifted_takeoff_curve_x[-1] + 5.0)
    y_takeoff = takeoff_line_slope * x_takeoff + takeoff_line_intercept

    # plot flight trajectory
    flight_xy = (flight_traj_x, flight_traj_y)

    return approach_xy, takeoff_xy, flight_xy


inputs = [Input('slope_angle', 'value'), Input('start_pos', 'value'),
          Input('approach_len', 'value'), Input('takeoff_angle', 'value')]


@app.callback(Output('slope-text', 'children'), [Input('slope_angle', 'value')])
def update_slope_text(slope_angle):
    slope_angle = float(slope_angle)
    return 'Slope Angle: {:0.0f} degrees'.format(slope_angle)


@app.callback(Output('takeoff-text', 'children'), [Input('takeoff_angle', 'value')])
def update_takeoff_text(takeoff_angle):
    takeoff_angle = float(takeoff_angle)
    return 'Takeoff Angle: {:0.0f} degrees'.format(takeoff_angle)


@app.callback(Output('my-graph', 'figure'), inputs)
def update_graph(slope_angle, start_pos, approach_len, takeoff_angle):

    slope_angle = float(slope_angle)
    start_pos = float(start_pos)
    approach_len = float(approach_len)
    takeoff_angle = float(takeoff_angle)

    jump_x, jump_y, traj_x, traj_y = make_jump(slope_angle, start_pos,
                                               approach_len, takeoff_angle)
    ap_xy, to_xy, fl_xy = create_plot_arrays(slope_angle, start_pos,
                                             approach_len, takeoff_angle,
                                             jump_x, jump_y, traj_x, traj_y)

    return {'data': [{'x': [0], 'y': [0], 'name': 'Slope Top'},
                     {'x': ap_xy[0], 'y': ap_xy[1], 'name': 'Approach'},
                     {'x': to_xy[0], 'y': to_xy[1], 'name': 'Takeoff'},
                     {'x': fl_xy[0], 'y': fl_xy[1], 'name': 'Flight'}],
            'layout': layout}

if __name__ == '__main__':
    app.run_server()
