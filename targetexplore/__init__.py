import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from .targets import Targets


def random_pointings(
    n,
    ra_lims=[0, 360],
    dec_lims=[-90, 90],
    seed=None,
    return_targets=False,
    **kwargs,
):
    np.random.seed(seed)
    ra = np.random.uniform(*ra_lims, n)
    dec = np.arcsin(
        np.random.uniform(*np.sin(np.array(dec_lims) * np.pi / 180), n)
    ) * (180 / np.pi)
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    if return_targets:
        return Targets(coords, **kwargs)
    return coords
