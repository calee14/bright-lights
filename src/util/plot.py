import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import mplfinance as mpf
import pandas as pd
from PIL import Image
import io

matplotlib.use("Agg")


def plot_recent_candlesticks(df, last_n_periods=60, filename=None, dark_mode=True):
    # Make sure datetime is the index and is datetime type
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    # Sort by datetime and get the most recent periods
    df = df.sort_index()
    df = df.tail(last_n_periods)

    if dark_mode:
        # Dark mode colors
        bg_color = "#1a1a1a"  # Deep dark gray (not pitch black)
        up_color = "#26a69a"  # Teal green for up candles
        down_color = "#ef5350"  # Red for down candles
        text_color = "#ffffff"  # White text
        grid_color = "#404040"  # Lighter gray for grid

        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",  # Start with dark base
            marketcolors=mpf.make_marketcolors(
                up=up_color,
                down=down_color,
                volume={"up": up_color, "down": down_color},
                edge="inherit",
                wick={"up": up_color, "down": down_color},
            ),
            gridstyle="--",
            gridcolor=grid_color,
            facecolor=bg_color,
            edgecolor=bg_color,
            figcolor=bg_color,
            rc={
                "grid.alpha": 0.3,
                "text.color": text_color,
                "axes.labelcolor": text_color,
                "xtick.color": text_color,
                "ytick.color": text_color,
                "axes.edgecolor": grid_color,
                "axes.facecolor": bg_color,
                "figure.facecolor": bg_color,
            },
        )
    else:
        # Original light mode
        up_color = "#089981"
        down_color = "#f33745"
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

    kwargs = {
        "type": "candle",
        "volume": True,
        "style": style,
        "ylabel": "Price",
        "ylabel_lower": "Volume",
        "figsize": (10, 6),
        "panel_ratios": (3, 1),
        "returnfig": True,
        "tight_layout": False,
        "volume_yscale": "log",
        "xrotation": 15,
    }

    # Create the plot
    fig, axes = mpf.plot(df, **kwargs)
    for ax in axes:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_ticks_position("right")

    if filename:
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=130,
            bbox_inches="tight",
            facecolor=bg_color if dark_mode else "white",
        )
        buf.seek(0)

        img = Image.open(buf)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        img.save(filename, "JPEG", quality=95, optimize=True)
        buf.close()
        plt.close(fig)

    return fig


def plot_recent_candlesticks_minimal(
    df, last_n_periods=60, filename=None, dark_mode=True
):
    """
    Ultra-optimized version with dark mode support
    """
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.tail(last_n_periods)

    if dark_mode:
        bg_color = "#1a1a1a"
        up_color = "#26a69a"
        down_color = "#ef5350"
        text_color = "#ffffff"
        grid_color = "#404040"

        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mpf.make_marketcolors(
                up=up_color,
                down=down_color,
                volume={"up": up_color, "down": down_color},
                edge="inherit",
                wick={"up": up_color, "down": down_color},
            ),
            gridstyle="--",
            gridcolor=grid_color,
            facecolor=bg_color,
            edgecolor=bg_color,
            figcolor=bg_color,
            rc={
                "grid.alpha": 0.3,
                "text.color": text_color,
                "axes.labelcolor": text_color,
                "xtick.color": text_color,
                "ytick.color": text_color,
                "axes.edgecolor": grid_color,
                "axes.facecolor": bg_color,
                "figure.facecolor": bg_color,
            },
        )
    else:
        up_color = "#089981"
        down_color = "#f33745"
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

    kwargs = {
        "type": "candle",
        "volume": True,
        "style": style,
        "ylabel": "Price",
        "ylabel_lower": "Volume",
        "figsize": (8, 5),
        "panel_ratios": (3, 1),
        "returnfig": True,
        "tight_layout": False,
        "volume_yscale": "log",
        "xrotation": 15,
    }

    fig, axes = mpf.plot(df, **kwargs)
    for ax in axes:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_ticks_position("right")

    if filename:
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=80,
            bbox_inches="tight",
            facecolor=bg_color if dark_mode else "white",
        )
        buf.seek(0)

        img = Image.open(buf)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        img.save(filename, "JPEG", quality=80, optimize=True)
        buf.close()
        plt.close(fig)

    return fig
