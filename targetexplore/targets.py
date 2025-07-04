import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from targetexplore.gaia import circles
from scipy.spatial import cKDTree
from targetexplore._utils.data import TELESCOPE_BAND as TB
from targetexplore._utils.data import DISC_DIAMETER as D
from targetexplore._utils.conversion import *
from targetexplore._utils.distance import *
from targetexplore._utils import calculate_density
from matplotlib import pyplot as plt


class Targets:
    """
    Represents a set of targets. Main inner class of the module

    ...

    Attributes
    ----------
    centers: SkyCoord
    radius: radius of list of radiuses for observations

    After self.query() (initialized to None)
        table: table of targets
        skycoord: skycoord vector with targets
        center: vector with nearest center for each target
        separation: angular separation to nearest target

    Methods
    -------

    info(additional=""): Prints the person's name and age.
    """
    def __init__(self, centers: SkyCoord, radius, query=True, L=1.35,
                 r_max=10000, dr=0.01, *args, **kwargs):
        """
        Initialize a Targets object

        Parameters:
            centers (Skycoord): SkyCoord coordinates of centers to point
            radius: either a single radius or a list, in arcminutes
            query (boolean): if true, immediately queries Gaia for the targets
            L (kpc, float): prior median distance of stars
        """

        self.centers = np.atleast_1d(centers).transform_to('icrs')
        self.centers.location = None  # makes sure it's barycentric
        self.centers.obstime = Time('J2016.0')

        self.center = None # for each point, the nearest center
        self.radius = np.array(radius)
        self.L = L
        self._r_max = r_max
        self._dr = dr

        self._center_tree = cKDTree(self.centers.cartesian.xyz.value.T)
        self.separation = None
        self.table = None
        self._posterior_dist = self._naive_dist = None

        if query: self.query(*args, **kwargs)

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
        assert self.table is not None, "run self.query() first"
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

        assert self.separation is not None, "run self._calculate_separations() first"
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

    def hist_parallax(self, ax=None):
        assert self.table is not None, "run self.query() first"
        plx = self.table["parallax"]/1000
        plx_err = self.table["parallax_error"]/1000
        if ax is None: ax = plt.gca()
        ax.hist(plx, bins=50, histtype="step")
        ax.plot(plx, 0*plx, '|', color='k', markersize=10)
        return ax

    @property
    def posterior_dist(self):
        if self._posterior_dist: return self._posterior_dist
        plx = self.table["parallax"]/1000
        plx_err = self.table["parallax_error"]/1000
        self._posterior_dist = posterior_distance(
            plx, plx_err, self.L, self._r_max, self._dr
        )
        return self.posterior_dist

    @property
    def naive_dist(self):
        if self._naive_dist: return self._naive_dist
        plx = self.table["parallax"]/1000
        plx_err = self.table["parallax_error"]/1000
        self._naive_dist = naive_distance(plx, plx_err)
        return self.naive_dist

    @property
    def M_naive(self):
        return self.table["phot_g_mean_mag"] - 5 * np.log10(self.naive_dist[0]) + 5

    @property
    def M_naive_err(self):
        safe_dist = np.where(self.naive_dist[0] != 0, self.naive_dist[0], 1e-6)
        return (5 / (np.log(10) * safe_dist)) * self.naive_dist[1]

    @property
    def M_bayes(self):
        return self.table["phot_g_mean_mag"] - 5 * np.log10(self.posterior_dist[0]) + 5

    @property
    def M_bayes_err(self):
        return (5 / (np.log(10) * self.posterior_dist[0])) * self.posterior_dist[1]

    def _hr_scatter(self, naive, bayes, ax, mask_parallax=False):
        color = self.table['phot_bp_mean_mag'] - self.table['phot_rp_mean_mag']

        M_naive, M_bayes = self.M_naive, self.M_bayes
        if mask_parallax:
            mask = (self.table['parallax'] > 0) & ~np.isnan(color) & ~np.isnan(M_bayes) & ~np.isnan(M_naive)
            color = color[mask]
            M_naive = M_naive[mask]; M_bayes = M_bayes[mask]

        if naive: ax.scatter(color, M_naive, s=1, alpha = .4)
        if bayes: ax.scatter(color, M_bayes, s=1, alpha = .4)
        ax.invert_yaxis()
        ax.set_xlabel('BP − RP')
        ax.set_ylabel('Absolute G magnitude (M_G)')
        ax.set_title('Hertzsprung–Russell Diagram')
        return ax

    def _hr_heatmap(self, naive, bayes, fig, ax):
        color = self.table['phot_bp_mean_mag'] - self.table['phot_rp_mean_mag']
        axl, axr = ax
        X_b, Y_b, density_bayes = calculate_density(
            color, self.M_bayes, color/100, self.M_bayes_err
        )
        X_n, Y_n, density_naive = calculate_density(
            color, self.M_naive, color/100, self.M_naive_err
        )
        naive = axl.contourf(X_n, Y_n, density_naive, levels=100, cmap="inferno")
        axl.set_xlabel("BP - RP")
        axl.set_ylabel("Absolute G magnitude")
        axl.set_title("HR using naive distance")
        axl.invert_yaxis()
        fig.colorbar(naive)

        bayes = axr.contourf(X_b, Y_b, density_bayes, levels=100, cmap="inferno")
        axr.set_xlabel("BP - RP")
        axr.set_title("HR using bayesian calculated distance")
        axr.invert_yaxis()
        fig.colorbar(bayes)

        fig.tight_layout()
        return fig, ax

    def hr(self,
           naive=True, bayes=True,
           heatmap=True,
           ax1 = None,
           fig2 = None, axs2 = None,
           mask_parallax = False):
        assert self.table is not None, "run self.query() first"
        if ax1 is None: _, ax1 = plt.subplots()
        ax1 = self._hr_scatter(naive, bayes, ax1, mask_parallax)
        if heatmap and (fig2 is None or axs2 is None):
            fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6))
        if heatmap:
            fig2, axs2 = self._hr_heatmap(naive, bayes, fig2, axs2)
        return ax1, fig2, axs2
