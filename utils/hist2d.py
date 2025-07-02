import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

from utils.clip_voronoi_2d import clip_voronoi


def _log_points(x, y, axis_scale):
    if axis_scale == "linear": return x, y, np.column_stack((x, y))
    if axis_scale == "log":
        mask = (x > 0) & (y > 0)
        x_valid = x[mask]
        y_valid = y[mask]
        log_x = np.log10(x_valid)
        log_y = np.log10(y_valid)
        points = np.column_stack((log_x, log_y))
        return log_x, log_y, points
    raise ValueError("linear-linear or log-log")

def hex_and_voronoi(x, y, **kwargs):
    fig, (hextile, voronoi) = plt.subplots(1, 2, figsize=(14, 6))

    axis_scale = kwargs.get("axis", "log")

    _ , hexhist = hex_hist(x, y, kwargs.get("gridsize"), hextile, axis_scale=axis_scale)
    _ , vorhist = voronoi_hist(x, y, kwargs.get("n_bins"), voronoi, ylabel=False, axis_scale=axis_scale)

    fig.colorbar(hexhist, ax=hextile, label="log(Count) per tile")
    fig.colorbar(vorhist, ax=voronoi, label="Count per tile")

    fig.tight_layout()
    plt.show()

def _ticks_n_title(x, y, ax=None, ylabel=True, lims=None):
    if ax is None: ax = plt.gca()
    # ticks_x = np.arange(np.floor(x.min()), np.ceil(x.max())+1)
    # ticks_y = np.arange(np.floor(y.min()), np.ceil(y.max())+1)
    # ax.set_xticks(ticks_x, [f"$10^{{{int(t)}}}$" for t in ticks_x])
    # ax.set_yticks(ticks_y, [f"$10^{{{int(t)}}}$" for t in ticks_y])
    if ylabel: ax.set_ylabel(r"Absolute G magnitude")
    ax.set_xlabel("BP - RP")

    if lims: ax.set_xlim([lims[0], lims[1]])
    if lims: ax.set_ylim([lims[2], lims[3]])
    ax.set_ylim([-2, 14])
    ax.set_xlim([0, 7])

    ax.invert_yaxis()

    return ax


def hex_hist(x, y, gridsize=30, ax=None, axis_scale="log"):
    log_x, log_y, _ = _log_points(x, y, axis_scale)
    if ax is None: ax = plt.gca() # I won't test this

    hexhist = ax.hexbin(log_x, log_y, gridsize=gridsize, cmap='inferno', bins='log')
    ax.set_title("Constant diameter histogram")

    return _ticks_n_title(log_x, log_y, ax), hexhist

def voronoi_hist(x, y, n_bins=150, ax=None, rs=420, ylabel=True, axis_scale="log"):
    log_x, log_y, points = _log_points(x, y, axis_scale)
    if ax is None: ax = plt.gca() # I won't test this

    kmeans = KMeans(n_clusters=n_bins, random_state=rs).fit(points)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    vor = Voronoi(centroids)
    regions, vertices = clip_voronoi(vor, radius=100)

    patches = []
    colors = []

    x_min, x_max = log_x.min(), log_x.max()
    y_min, y_max = log_y.min(), log_y.max()
    for point_idx, region in enumerate(regions):
        polygon = vertices[region]

        # For simplicity, just discard polygons completely outside the view
        if (polygon[:,0].max() < x_min or polygon[:,0].min() > x_max or
            polygon[:,1].max() < y_min or polygon[:,1].min() > y_max):
            continue

        patches.append(Polygon(polygon, closed=True))
        colors.append(counts[point_idx])

    norm = mpl.colors.Normalize(vmin=np.min(colors), vmax=np.max(colors))
    cmap = plt.get_cmap("inferno")

    collection = PatchCollection(patches, cmap=cmap, norm=norm)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    ax.set_title("Voronoi histogram")

    return _ticks_n_title(log_x, log_y, ax, ylabel), collection
