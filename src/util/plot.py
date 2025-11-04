import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import mplfinance as mpf
import pandas as pd
from PIL import Image
import io

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

    # OPTIMIZATION 1: Reduce figure size (smaller dimensions = fewer tokens)
    # Changed from (12, 8) to (10, 6) - still readable but 37.5% fewer pixels
    kwargs = {
        "type": "candle",
        "volume": True,
        "style": style,
        "ylabel": "Price",
        "ylabel_lower": "Volume",
        "figsize": (10, 6),  # Reduced from (12, 8)
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
        # OPTIMIZATION 2: Reduce DPI from 130 to 95
        # This maintains readability while reducing file size significantly
        # 95 DPI gives you 950x570 pixels (vs 1560x1040 at 130 DPI)

        # Save to temporary buffer first
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
        buf.seek(0)

        # OPTIMIZATION 3: Compress the image further
        img = Image.open(buf)

        # Optional: Convert to RGB if needed (removes alpha channel)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Save with optimized compression
        # Quality 85 is a sweet spot - minimal visual difference but good compression
        img.save(filename, "JPEG", quality=85, optimize=True)

        buf.close()
        plt.close(fig)

    return fig


def plot_recent_candlesticks_minimal(df, last_n_periods=60, filename=None):
    """
    Ultra-optimized version for maximum token savings.
    Use this if you want even more aggressive optimization.
    ~60-70% smaller file size than original.
    """
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.tail(last_n_periods)

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

    # Even smaller figure size
    kwargs = {
        "type": "candle",
        "volume": True,
        "style": style,
        "ylabel": "Price",
        "ylabel_lower": "Volume",
        "figsize": (8, 5),  # Further reduced
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
        # Lower DPI for smaller file
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)

        img = Image.open(buf)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Slightly lower quality but still very readable for charts
        img.save(filename, "JPEG", quality=80, optimize=True)

        buf.close()
        plt.close(fig)

    return fig
