"""One-call professional defaults for every notebook: consistent plots, quiet warnings, readable tables."""
import warnings


def setup(dpi=110):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    warnings.filterwarnings("ignore")
    sns.set_theme(context="notebook", style="whitegrid", palette="deep", font_scale=0.95)
    plt.rcParams.update({
        "figure.dpi": dpi, "savefig.dpi": 200, "figure.autolayout": False,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.35, "legend.frameon": False, "image.cmap": "viridis",
    })
    pd.options.display.float_format = "{:,.4g}".format
    pd.options.display.max_columns = 20
