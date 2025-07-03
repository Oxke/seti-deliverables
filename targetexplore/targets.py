import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from targetexplore.gaia import circles
from scipy.spatial import cKDTree
from targetexplore._utils.data import TELESCOPE_BAND as TB
from targetexplore._utils.data import DISC_DIAMETER as D
from targetexplore._utils.conversion import *
from matplotlib import pyplot as plt


class Targets:
    """
    Represents a set of targets. Main inner class of the module

    ...

    Attributes
    ----------
    centers: SkyCoord
    radius: radius of list of radiuses for observations
    targets: table of targets (after self.query(), otherwise None)

    Methods
    -------

    info(additional=""): Prints the person's name and age.
    """
    def __init__(self, centers: SkyCoord, radius, query=True):
        """
        Initialize a Targets object

        Parameters:
            centers (Skycoord): SkyCoord coordinates of centers to point
            radius: either a single radius or a list, in arcminutes
            query (boolean): if true, immediately queries Gaia for the targets
        """

        self.centers = np.atleast_1d(centers).transform_to('icrs')
        self.centers.location = None  # makes sure it's barycentric
        self.centers.obstime = Time('J2016.0')

        self.center = None # for each point, the nearest center
        self.radius = np.array(radius)

        self._center_tree = cKDTree(self.centers.cartesian.xyz.value.T)
        self.separation = None
        self.table = None

        if query: self.query()

    def query(self, calculate_separations=True, *args, **kwargs):
        """
        set calculate_separations to False to only query, without updating the
        self.separation variable
        """
        self.table, job = circles(self.centers, self.radius / 60,
                                    *args, **kwargs) # degrees
        if calculate_separations:
            self.separation, self.center = self._calculate_separations()
        return job

    @property
    def skycoord(self):
        if self.table is None: return None
        return SkyCoord(ra=self.table['ra'], dec=self.table['dec'], frame='icrs')

    def _calculate_separations(self):
        sources_xyz = self.skycoord.cartesian.xyz.value.T
        dists, idxs = self._center_tree.query(sources_xyz, k=1)
        ang_sep = 2 * np.arcsin(dists / 2) * u.rad
        return ang_sep.to(u.arcmin), self.centers[idxs]

    def hist_separation(self,
                        highlight=None, # if set to true, needs telescope and
                                        # band, otherwise can be set to (lo, hi)
                                        # in arcmin
                        disc_diameter=None, # if set, shows second axis with frequency
                        telescope=None,
                        band=None,
                        title=None,
                        savefig_name=None,
                        *args, **kwargs): # passed to bar plot

        lo, hi = np.inf, -np.inf # not highlighting anything
        if highlight is True:
            assert f"{telescope}_{band}" in TB, "band not known, please \
                explicit it by setting highlight=(low arcmin, high arcmin)"
            disc_diameter = D[telescope]
            hi, lo = vfreq_to_angle(TB[f"{telescope}_{band}"], disc_diameter)
            lo, hi = lo/2, hi/2
        elif highlight: lo, hi = highlight
        counts, bin_edges = np.histogram(self.separation, bins=100)

        for i in range(1, len(counts)):
            counts[i] += counts[i-1]
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        colors = ['crimson' if lo <= c.value <= hi else '#007847' for c in bin_centers]

        plt.figure(figsize=(8, 5))
        plt.bar(
            bin_centers*2,
            counts,
            width=2*np.diff(bin_edges),
            color=colors,
            align='center',
            *args, **kwargs
        )
        plt.xlabel("Field of view (arcminutes)")
        plt.ylabel("Number of Gaia DR3 targets")
        if title: plt.title(title)
        plt.gca().invert_xaxis()

        if disc_diameter:
            secax = plt.gca().secondary_xaxis(
                'top',
                functions=(
                    lambda f: freq_to_angle(f, disc_diameter),
                    lambda t: angle_to_freq(t, disc_diameter)
                )
            )
            secax.set_xlabel("Frequency observed (GHz)")

            (flo, fhi), d = TB[f'{telescope}_{band}'], D[telescope]
            m1 = (2*freq_to_angle(flo, d) + freq_to_angle(fhi, d))/3
            m2 = (freq_to_angle(flo, d) + 2*freq_to_angle(fhi, d))/3
            secax.set_xticks([flo, angle_to_freq(m1, d), angle_to_freq(m2, d),
                              fhi, 1.5*fhi, 3*fhi, 1000])

        plt.tight_layout()
        if savefig_name: plt.savefig(savefig_name)
        plt.show()


