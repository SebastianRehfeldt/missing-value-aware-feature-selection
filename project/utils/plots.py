import numpy as np
import missingno
from project.utils import Data


# https://github.com/ResidentMario/missingno
def sort_by_class(data):
    # SORT X AND Y BY CLASS TO CHECK IF NANS ARE EQUALLY DISTRIBUTED
    sorted_indices = np.argsort(data.y)
    X = data.X.iloc[sorted_indices]
    y = data.y[sorted_indices]
    return Data(X, y, data.f_types, data.l_type, data.shape)


def plot_nans_by_class(data):
    # PLOTS BARS FOR FEATURES WHERE HOLES ARE NANS
    X = sort_by_class(data).X
    return missingno.matrix(X)


def plot_nan_percentage(data):
    # PLOTS PERCENTAGE OF NANS USING BARS
    return missingno.bar(data.X)


def plot_nan_correlation(data):
    # PLOTS PERCENTAGE OF NANS USING BARS
    """
        -1 (if one variable appears the other definitely does not)
         0 (variables appearing or not appearing have no effect on one another)
         1 (if one variable appears the other definitely also does).
    """
    return missingno.heatmap(data.X)


def plot_nan_dendogram(data):
    # PLOTS PERCENTAGE OF NANS USING BARS
    """
        Cluster leaves which linked together at a distance of zero 
        fully predict one another's presence
    """
    return missingno.dendrogram(data.X)


def show_boxplots(data, features=None):
    features = data.X.columns if features is None else features
    df = data.X[features]
    df["class"] = data.y
    return df.boxplot(grid=False, by="class")


def show_scatter_plots(data, features):
    classes = np.unique(data.y)
    colors = "rgbcmyk" * 10
    for i, c in enumerate(classes):
        d = data.X[features][data.y == c]
        if i == 0:
            ax = d.plot.scatter(
                x=features[0],
                y=features[1],
                color=colors[i],
                label=c,
            )
        else:
            d.plot.scatter(
                x=features[0],
                y=features[1],
                color=colors[i],
                label=c,
                ax=ax,
            )
    return ax


def show_correlation(data, features=None):
    f = data.X.columns if features is None else features
    a = data.X[f].corr().style.background_gradient().set_precision(2)
    return a
