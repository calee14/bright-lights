import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import mplfinance as mpf
import pandas as pd

matplotlib.use("Agg")


def plot_recent_candlesticks(df, last_n_periods=60, filename=None):
    # Make sure datetime is the index and is datetime type
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)

    # Sort by datetime and get the most recent periods
    df = df.sort_index()
    df = df.tail(last_n_periods)

    up_color = "#089981"
    down_color = "#f33745"

    # Define style
    style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(
            up=up_color,
            down=down_color,
            volume={"up": up_color, "down": down_color},
            edge="inherit",
        ),
        gridstyle="--",
        gridcolor="grey",
        rc={"grid.alpha": 0.3},
    )

    # Configure the plot
    kwargs = {
        "type": "candle",
        "volume": True,
        "style": style,
        "ylabel": "Price",
        "ylabel_lower": "Volume",
        "figsize": (12, 8),
        "panel_ratios": (3, 1),
        "returnfig": True,
        "tight_layout": False,
        "volume_yscale": "log",  # Enable logarithmic scale for volume
        "xrotation": 15,  # Rotate x-axis labels for better readability
    }

    # Create the plot
    fig, axes = mpf.plot(df, **kwargs)

    for ax in axes:
        ax.yaxis.tick_right()  # Move ticks to the right
        ax.yaxis.set_label_position("right")  # Move label to the right
        ax.yaxis.set_ticks_position("right")  # Ensure ticks are on the right side

    if filename:
        fig.savefig(filename, dpi=130, bbox_inches="tight")
        plt.close(fig)

    return fig
