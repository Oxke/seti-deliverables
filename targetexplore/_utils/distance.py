#!/usr/bin/env python
import numpy as np
from scipy.integrate import simpson
from scipy.stats import norm


def posterior_distance(parallax, parallax_error, L=1.35, r_max=10000.0, dr=0.01):
    """Computes the posterior distance PDF for a given parallax using Bailer-Jones method."""
    r = np.arange(dr, r_max, dr)

    var = parallax_error**2
    likelihood = norm.pdf(parallax, loc=1/r, scale=parallax_error) # p( parallax | r)
    prior = r**2 * np.exp(-r / L)
    posterior = likelihood * prior
    norm_posterior = posterior / simpson(posterior, r)

    cdf = np.cumsum(norm_posterior) * dr
    median = r[np.searchsorted(cdf, 0.5)]
    lo = r[np.searchsorted(cdf, 0.16)]
    hi = r[np.searchsorted(cdf, 0.84)]

    return median, (hi - lo) / 2
posterior_distance = np.vectorize(posterior_distance)

def naive_distance(parallax, parallax_error):
    return 1/parallax, parallax_error / parallax**2
naive_distance = np.vectorize(naive_distance)
