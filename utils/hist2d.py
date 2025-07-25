import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

from utils.clip_voronoi_2d import clip_voronoi


def _log_points(x, y, axis_scale):
    if axis_scale == "linear":
        return x, y, np.column_stack((x, y))
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

    _, hexhist = hex_hist(x, y, kwargs.get("gridsize"), hextile, axis_scale=axis_scale)
    _, vorhist = voronoi_hist(
        x, y, kwargs.get("n_bins"), voronoi, ylabel=False, axis_scale=axis_scale
    )

    fig.colorbar(hexhist, ax=hextile, label="log(Count) per tile")
    fig.colorbar(vorhist, ax=voronoi, label="Count per tile")

    fig.tight_layout()
    plt.show()


def _ticks_n_title(x, y, ax=None, ylabel=True, lims=None):
    if ax is None:
        ax = plt.gca()
    # ticks_x = np.arange(np.floor(x.min()), np.ceil(x.max())+1)
    # ticks_y = np.arange(np.floor(y.min()), np.ceil(y.max())+1)
    # ax.set_xticks(ticks_x, [f"$10^{{{int(t)}}}$" for t in ticks_x])
    # ax.set_yticks(ticks_y, [f"$10^{{{int(t)}}}$" for t in ticks_y])
    if ylabel:
        ax.set_ylabel(r"Absolute G magnitude")
    ax.set_xlabel("BP - RP")

    if lims:
        ax.set_xlim([lims[0], lims[1]])
    if lims:
        ax.set_ylim([lims[2], lims[3]])
    ax.set_ylim([-2, 14])
    ax.set_xlim([0, 7])

    ax.invert_yaxis()

    return ax


def hex_hist(x, y, gridsize=30, ax=None, axis_scale="log"):
    log_x, log_y, _ = _log_points(x, y, axis_scale)
    if ax is None:
        ax = plt.gca()  # I won't test this

    hexhist = ax.hexbin(log_x, log_y, gridsize=gridsize, cmap="inferno", bins="log")
    ax.set_title("Constant diameter histogram")

    return _ticks_n_title(log_x, log_y, ax), hexhist


def voronoi_hist(x, y, n_bins=150, ax=None, rs=420, ylabel=True, axis_scale="log"):
    log_x, log_y, points = _log_points(x, y, axis_scale)
    if ax is None:
        ax = plt.gca()  # I won't test this

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
        if (
            polygon[:, 0].max() < x_min
            or polygon[:, 0].min() > x_max
            or polygon[:, 1].max() < y_min
            or polygon[:, 1].min() > y_max
        ):
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


def dr_snr(signals):
    # Extract relevant columns
    dr = signals[:, 1]
    snr = signals[:, 2]
    detected = signals[:, 4]

    # Define bin edges
    dr_bins = np.linspace(dr.min(), dr.max(), 50)  # linear bins
    snr_bins = np.logspace(np.log10(snr.min()), np.log10(snr.max()), 50)  # log bins

    # Compute 2D histogram: sum of detections per bin
    det_sum, _, _ = np.histogram2d(dr, snr, bins=[dr_bins, snr_bins], weights=detected)

    # Compute 2D histogram: count of total points per bin
    det_count, _, _ = np.histogram2d(dr, snr, bins=[dr_bins, snr_bins])

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        detection_rate = det_sum / det_count
        # detection_rate[np.isnan(detection_rate)] = 0  # Fill empty bins with 0

    # Plotting
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(dr_bins, snr_bins)

    # Transpose because histogram2d returns (y_bins, x_bins)
    pcm = plt.pcolormesh(X, Y, detection_rate.T, cmap="viridis", shading="auto")

    plt.xlim(dr_bins[0], dr_bins[-1])
    plt.ylim(snr_bins[0], snr_bins[-1])
    plt.xscale("linear")
    plt.yscale("log")
    plt.colorbar(pcm, label="Detection Rate (%)")
    plt.xlabel("dr")
    plt.ylabel("snr (log scale)")
    plt.title("Detection Rate as Function of dr and snr")
    plt.tight_layout()


def dr_width(signals, xscale="linear", yscale="linear", xlim=None, ylim=None):
    # Extract relevant columns
    dr = signals[:, 1]
    width = signals[:, 3]
    detected = signals[:, 4]

    # Define bins for dr and width (linear for both)
    dr_bins = np.linspace(*(xlim if xlim else (dr.min(), dr.max())), 50)
    width_bins = np.linspace(*(ylim if ylim else (width.min(), width.max())), 50)

    # Compute detection sum and count per bin
    det_sum, _, _ = np.histogram2d(
        dr, width, bins=[dr_bins, width_bins], weights=detected
    )
    det_count, _, _ = np.histogram2d(dr, width, bins=[dr_bins, width_bins])

    # Compute detection rate (avoid divide-by-zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        detection_rate = det_sum / det_count
        # detection_rate[np.isnan(detection_rate)] = 0

    # Plotting
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(dr_bins, width_bins)
    pcm = plt.pcolormesh(X, Y, detection_rate.T, cmap="viridis", shading="auto")

    plt.xscale(xscale)
    plt.yscale(yscale)

    plt.xlabel("dr")
    plt.ylabel("width")
    plt.title("Detection Rate as Function of dr and width")
    plt.colorbar(pcm, label="Detection Rate")
    plt.xlim(dr_bins[0], dr_bins[-1])
    plt.ylim(width_bins[0], width_bins[-1])
    plt.tight_layout()

def plot_detection_rate(signals, x_idx, y_idx, n_x=50, n_y=50, xscale="linear", yscale="linear",
                        xlim=None, ylim=None, xlabel="", ylabel="", title="", cmap="viridis"):
    x = signals[:, x_idx]
    y = signals[:, y_idx]
    detected = signals[:, 4]

    # Define bins
    if xscale == "log":
        x_bins = np.logspace(*np.log10(xlim if xlim else (x.min(), x.max())), n_x)
    else:
        x_bins = np.linspace(*(xlim if xlim else (x.min(), x.max())), n_x)

    if yscale == "log":
        y_bins = np.logspace(*np.log10(ylim if ylim else (y.min(), y.max())), n_y)
    else:
        y_bins = np.linspace(*(ylim if ylim else (y.min(), y.max())), n_y)

    # Compute detection rate
    det_sum, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=detected)
    det_count, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    with np.errstate(divide="ignore", invalid="ignore"):
        detection_rate = det_sum / det_count

    # Plot
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(x_bins, y_bins)
    pcm = plt.pcolormesh(X, Y, detection_rate.T, cmap=cmap, shading="auto")
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(pcm, label="Detection Rate")
    plt.xlim(x_bins[0], x_bins[-1])
    plt.ylim(y_bins[0], y_bins[-1])
    plt.tight_layout()

def snr_width(signals, swap_axes=False, xlim=None, ylim=None, n_x=50, n_y=50, cmap="YlOrRd"):
    x_idx, y_idx = 2, 3
    xscale, yscale = "log", "linear"
    xlabel, ylabel = "SNR", "width"
    if swap_axes:
        x_idx, y_idx = y_idx, x_idx
        xscale, yscale = yscale, xscale
        xlabel, ylabel = ylabel, xlabel
    plot_detection_rate(
        signals,
        x_idx=x_idx,
        y_idx=y_idx,
        n_x=n_x,
        n_y=n_y,
        xscale=xscale,
        yscale=yscale,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        title="Detection Rate as Function of snr and width"
    )
