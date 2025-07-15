import numpy as np
from scipy.stats import multivariate_normal


def calculate_density(colors, magnitudes, color_err, mag_err):
    x_min, x_max = colors.min(), colors.max()
    y_min, y_max = 1, 10
    x = np.linspace(x_min, x_max, 300)
    y = np.linspace(y_max, y_min, 300)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    density = np.zeros_like(X)

    for cx, cy, sx, sy in zip(colors, magnitudes, color_err, mag_err):
        if not (np.isfinite(cx) and np.isfinite(cy) and sx > 0 and sy > 0):
            continue
        cov = [[sx**2, 0], [0, sy**2]]
        rv = multivariate_normal(mean=[cx, cy], cov=cov)
        density += rv.pdf(pos)

    return X, Y, density / np.max(density)
