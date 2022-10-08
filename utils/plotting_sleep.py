import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime

def plot_sleep_2D(garmin_sleep_df, 
                    plot_pred = False, plot_schedule_change = False, 
                    pred_sleep_df = None, start_shift_date = None):
    """read epochs, heart rates and sleep data for one subject

    Parameters
    ----------
    garmin_sleep_df: pd.DataFrame 
    plot_pred: boolean
    plot_schedule_change: boolean
    pred_sleep_df: pd.DataFrame
    start_shift_date: str

    Returns
    -------
    plotly figure shown

    """
    fig = go.Figure()
    if plot_pred:
        fig.add_trace(go.Scatter(x=pred_sleep_df["Time_In_Day"],
                                 y=pred_sleep_df["Date"],
                                 mode='markers',
                                 marker_symbol='square',
                                 marker=dict(
            size=6.5, color="salmon", opacity=0.7),
            name="Imputed Sleep Labels"
        ))
    fig.add_trace(go.Scatter(x=garmin_sleep_df["Time_In_Day"],
                            y=garmin_sleep_df["Date"],
                            mode='markers',
                            marker_symbol='square',
                            marker=dict(
        size=6.5, color="royalblue", opacity=0.7),
        name="Garmin Sleep Labels"
    ))


    if plot_schedule_change:
        fig.add_trace(go.Scatter(x=[0, 24],
                                y=[start_shift_date for i in range(2)],
                                mode='lines',
                                line=dict(color='firebrick', width=4, dash='dot'),
                                name="Shift Start Date = {}".format(
            start_shift_date.strftime("%Y-%m-%d"))
        ))

    fig.update_xaxes(title_text="Time in day in hours", range=[0, 24])

    fig.update_yaxes(title_text="Date")

    h = 900
    w = 700

    # set figure size and theme
    # set figure size and theme
    fig.update_layout(height=h,
                        width=w,
                        template='simple_white',
                        title_text="Comparing Sleep Labels from Garmin and from Imputation",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=0.99,
                            xanchor="right",
                            x=1
                        ),
                        yaxis=dict(autorange="reversed"),
                        xaxis = dict(
                                    tickmode = 'array',
                                    tickvals = [0, 6, 12, 18, 24],
                                    ticktext = ['12 AM', '6 AM', '12 PM', '18 PM', '24 AM']
                                )
                        )
    config = {
        'toImageButtonOptions': {
            'format': 'png',  # one of png, svg, jpeg, webp
            'filename': 'large_figure',
            'height': h,
            'width': w,
            'scale': 12  # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    fig.show(config=config)
