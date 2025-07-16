import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
from .gaia import circles
from scipy.spatial import cKDTree
from ._utils.data import TELESCOPE_BAND as TB
from ._utils.data import DISC_DIAMETER as D
from ._utils.conversion import *
from ._utils.distance import *
from ._utils import calculate_density
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

    def __init__(
        self,
        centers: SkyCoord,
        radius=0,
        telescope=None,
        band=None,
        query=True,
        L=1.35,
        r_max=10000,
        dr=0.01,
        *args,
        **kwargs,
    ):
        """
        Initialize a Targets object

        Parameters:
            centers (Skycoord): SkyCoord coordinates of centers to point
            radius: either a single radius or a list, in arcminutes
            query (boolean): if true, immediately queries Gaia for the targets
            L (kpc, float): prior median distance of stars
        """

        self.centers = np.atleast_1d(centers).transform_to("icrs")
        self.centers.location = None  # makes sure it's barycentric
        self.centers.obstime = Time("J2016.0")

        self.center = None  # for each point, the nearest center
        self._max_radius = None  # used for the actual visualizations
        if radius == 0 and telescope is not None and band is not None:
            assert f"{telescope}_{band}" in TB, "band not known, please set the radius"
            disc_diameter = D[telescope]
            self._max_radius = freq_to_angle(
                TB[f"{telescope}_{band}"][1], disc_diameter
            )
            radius = np.ceil(self._max_radius)
        assert radius > 0, "either set radius os telescope and band"
        self.radius = np.array(radius)
        self.L = L
        self._r_max = r_max
        self._dr = dr

        self._center_tree = cKDTree(self.centers.cartesian.xyz.value.T)
        self.separation = None
        self._table = None
        self._posterior_dist = self._naive_dist = None

        self.telescope = telescope
        self.band = band

        if query:
            self.query(*args, **kwargs)

    @property
    def table(self):
        if self._max_radius is None or self.separation is None:
            return self._table
        return self._table[self.separation.value < self._max_radius]

    @table.setter
    def table(self, new_table):
        self._table = new_table

    def query(self, calculate_separations=True, *args, **kwargs):
        """
        set calculate_separations to False to only query, without updating the
        self.separation variable
        """
        self.table, self._query_job, self._query = circles(
            self.centers, self.radius / 60, *args, **kwargs
        )  # degrees
        if calculate_separations:
            self.separation, self.center = self._calculate_separations()
        return self._query_job

    @property
    def skycoord(self):
        if self.table is None:
            return None
        return SkyCoord(ra=self.table["ra"], dec=self.table["dec"], frame="icrs")

    def _calculate_separations(self):
        assert self.table is not None, "run self.query() first"
        sources_xyz = self.skycoord.cartesian.xyz.value.T
        dists, idxs = self._center_tree.query(sources_xyz, k=1)
        ang_sep = 2 * np.arcsin(dists / 2) * u.rad
        return ang_sep.to(u.arcmin), self.centers[idxs]

    def hist_separation(
        self,
        highlight=None,  # if set to true, needs telescope and
        # band, otherwise can be set to (lo, hi)
        # in arcmin
        disc_diameter=None,  # if set, shows second axis with frequency
        telescope=None,
        band=None,
        title=None,
        savefig_name=None,
        *args,
        **kwargs,
    ):  # passed to bar plot

        assert self.separation is not None, "run self._calculate_separations() first"
        lo, hi = np.inf, -np.inf  # not highlighting anything
        if telescope is None:
            telescope = self.telescope
        if band is None:
            band = self.band
        if highlight is True or (
            highlight is None and telescope is not None and band is not None
        ):
            assert (
                f"{telescope}_{band}" in TB
            ), "band not known, please \
                explicit it by setting highlight=(low arcmin, high arcmin)"
            disc_diameter = D[telescope]
            hi, lo = vfreq_to_angle(TB[f"{telescope}_{band}"], disc_diameter)
            lo, hi = lo / 2, hi / 2
        elif highlight:
            lo, hi = highlight
        counts, bin_edges = np.histogram(self.separation, bins=100)

        for i in range(1, len(counts)):
            counts[i] += counts[i - 1]
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        colors = ["crimson" if lo <= c.value <= hi else "#007847" for c in bin_centers]

        plt.figure(figsize=(8, 5))
        plt.bar(
            bin_centers * 2,
            counts,
            width=2 * np.diff(bin_edges),
            color=colors,
            align="center",
            *args,
            **kwargs,
        )
        plt.xlabel("Field of view (arcminutes)")
        plt.ylabel("Number of Gaia DR3 targets")
        if title:
            plt.title(title)
        plt.gca().invert_xaxis()

        if disc_diameter:
            secax = plt.gca().secondary_xaxis(
                "top",
                functions=(
                    lambda f: freq_to_angle(f, disc_diameter),
                    lambda t: angle_to_freq(t, disc_diameter),
                ),
            )
            secax.set_xlabel("Frequency observed (GHz)")

            (flo, fhi), d = TB[f"{telescope}_{band}"], D[telescope]
            m1 = (2 * freq_to_angle(flo, d) + freq_to_angle(fhi, d)) / 3
            m2 = (freq_to_angle(flo, d) + 2 * freq_to_angle(fhi, d)) / 3
            secax.set_xticks(
                [
                    angle_to_freq(self.radius * 2, d),
                    flo,
                    angle_to_freq(m1, d),
                    angle_to_freq(m2, d),
                    fhi,
                    1.5 * fhi,
                    3 * fhi,
                    1000,
                ]
            )

        plt.tight_layout()
        if savefig_name:
            plt.savefig(savefig_name)
        plt.show()

    def hist_parallax(self, ax=None, markers=True, *args, **kwargs):
        assert self.table is not None, "run self.query() first"
        plx = self.table["parallax"] / 1000
        plx_err = self.table["parallax_error"] / 1000
        if ax is None:
            ax = plt.gca()
        ax.hist(plx, bins=50, histtype="step", *args, **kwargs)
        if markers:
            ax.plot(plx, 0 * plx, "|", color="k", markersize=10)
        ax.set_xlabel("parallax")
        ax.set_ylabel("number of targets")
        return ax

    def hist_distance(
        self,
        method="gaia",
        ax=None,
        markers=True,
        bins=1 / 30,
        histtype="step",
        mask_parallax=False,
        *args,
        **kwargs,
    ):  # default plots gaia distances
        """
        the `bins` attribute specifies both how the bins are made (binning by equal area
        or equal width) and how many bins.
        setting a fraction 1/n makes bins that group every n items in the same bin.
        setting an integer n makes n bins of equal width
        """
        assert self.table is not None, "run self.query() first"
        if ax is None:
            ax = plt.gca()
        dists = None
        if method == "gaia":
            dists = self.gaia_dist[0]
        elif method == "naive":
            dists = self.naive_dist[0]
        elif method == "bayes":
            dists = self.posterior_dist[0]
        assert dists is not None, "method must be in ['gaia', 'naive', 'bayes']"
        if mask_parallax:
            mask = (self.table["parallax"] > 0) & ~np.isnan(dists)
            dists = dists[mask]
        if isinstance(bins, float):
            assert (
                1 / bins % 1 == 0
            ), "if floating point, `bins` must be of the form 1/n"
            q = int(1 / bins)
            bins = np.quantile(dists, np.linspace(0, 1, q + 1))
        ax.hist(dists, bins=bins, density=True, histtype=histtype, *args, **kwargs)
        if markers:
            ax.plot(dists, 0 * dists, "|", color="k", markersize=10)
        ax.set_xlabel("distance (pc)")
        ax.set_ylabel("number of targets")
        return ax

    @property
    def gaia_dist(self):
        err = (
            self.table["distance_gspphot_upper"] - self.table["distance_gspphot_lower"]
        ) / 2
        return np.array(self.table["distance_gspphot"]), np.array(err)

    @property
    def posterior_dist(self):
        if self._posterior_dist:
            return self._posterior_dist
        plx = self.table["parallax"] / 1000
        plx_err = self.table["parallax_error"] / 1000
        self._posterior_dist = posterior_distance(
            plx, plx_err, self.L, self._r_max, self._dr
        )
        return self.posterior_dist

    @property
    def naive_dist(self):
        if self._naive_dist:
            return self._naive_dist
        plx = self.table["parallax"] / 1000
        plx_err = self.table["parallax_error"] / 1000
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

    @property
    def M_gaia(self):
        return self.table["abs_g_mag"]

    @property
    def M_gaia_err(self):
        return self.table["abs_g_mag_error"]

    @property
    def color(self):
        return self.table["bp_rp"]

    @property
    def color_err(self):
        return self.table["bp_rp_error"]

    def __len__(self):
        return len(self.table)

    def _hr_scatter(self, naive, bayes, gaia, ax, mask_parallax=False):
        mask = np.ones_like(self.table["parallax"].data, dtype=bool)
        if mask_parallax:
            mask = (self.table["parallax"] > 0) & ~np.isnan(self.color)
            if naive:
                mask &= ~np.isnan(self.M_naive)
            if bayes:
                mask &= ~np.isnan(self.M_bayes)
            if gaia:
                mask &= ~np.isnan(self.M_gaia)

        if naive:
            ax.scatter(
                self.color[mask],
                self.M_naive[mask],
                s=max(1000 / len(self), 5),
                alpha=0.4,
                label="naive distance estimate",
            )
        if bayes:
            ax.scatter(
                self.color[mask],
                self.M_bayes[mask],
                s=max(1000 / len(self), 5),
                alpha=0.4,
                label="bayesian distance estimate",
            )
        if gaia:
            ax.scatter(
                self.color[mask],
                self.M_gaia[mask],
                s=max(1000 / len(self), 5),
                alpha=0.4,
                label="distance as computed in gaia catalog",
            )
        ax.invert_yaxis()
        ax.set_xlabel("BP − RP")
        ax.set_ylabel("Absolute G magnitude (M_G)")
        ax.set_title("Hertzsprung–Russell Diagram")
        ax.legend()
        return ax

    def _hr_heatmap_gaia(self, fig, ax, mask_parallax, cmap):
        mask = np.ones_like(self.table["parallax"].data, dtype=bool)
        if mask_parallax:
            mask = (
                (self.table["parallax"] > 0)
                & ~np.isnan(self.color)
                & ~np.isnan(self.M_gaia)
                & ~np.isnan(self.M_gaia_err)
            )
        X, Y, density = calculate_density(
            self.color, self.M_gaia, self.color_err, self.M_gaia_err
        )
        gaia_hr = ax.contourf(X, Y, density, levels=100, cmap=cmap)
        ax.set_xlabel("BP - RP")
        ax.set_ylabel("Absolute G magnitude")
        ax.set_title("HR using distance as computed in GAIA catalog")
        ax.invert_yaxis()
        fig.colorbar(gaia_hr)

        return fig, ax

    def _hr_heatmap(self, naive, bayes, fig, ax, mask_parallax, cmap):
        if naive and bayes:
            axl, axr = ax
        elif naive:
            axl = ax
        elif bayes:
            axr = ax

        mask = np.ones_like(self.table["parallax"].data, dtype=bool)
        if mask_parallax:
            mask = (self.table["parallax"] > 0) & ~np.isnan(self.color)
            if naive:
                mask &= ~np.isnan(self.M_naive) & ~np.isnan(self.M_naive_err)
            if bayes:
                mask &= ~np.isnan(self.M_bayes) & ~np.isnan(self.M_bayes_err)

        if naive:
            X_n, Y_n, density_naive = calculate_density(
                self.color[mask],
                self.M_naive[mask],
                self.color_err[mask],
                self.M_naive_err[mask],
            )
            naive = axl.contourf(X_n, Y_n, density_naive, levels=100, cmap=cmap)
            axl.set_xlabel("BP - RP")
            axl.set_ylabel("Absolute G magnitude")
            axl.set_title("HR using naive distance")
            axl.invert_yaxis()
            fig.colorbar(naive)

        if bayes:
            X_b, Y_b, density_bayes = calculate_density(
                self.color[mask],
                self.M_bayes[mask],
                self.color_err[mask],
                self.M_bayes_err[mask],
            )
            bayes = axr.contourf(X_b, Y_b, density_bayes, levels=100, cmap=cmap)
            axr.set_xlabel("BP - RP")
            axr.set_title("HR using bayesian calculated distance")
            axr.invert_yaxis()
            fig.colorbar(bayes)

        fig.tight_layout()
        return fig, ax

    def hr(
        self,
        gaia=True,
        naive=True,
        bayes=True,
        heatmap=True,
        ax1=None,
        fig2=None,
        axs2=None,
        fig3=None,
        ax3=None,
        mask_parallax=False,
        cmap="inferno",
    ):
        assert self.table is not None, "run self.query() first"
        if ax1 is None:
            _, ax1 = plt.subplots()
        ax1 = self._hr_scatter(naive, bayes, gaia, ax1, mask_parallax)
        if heatmap and (fig2 is None or axs2 is None):
            n = naive + bayes
            fig2, axs2 = plt.subplots(1, n, figsize=(7 * n, 6))
        if heatmap:
            fig2, axs2 = self._hr_heatmap(naive, bayes, fig2, axs2, mask_parallax, cmap)
        if gaia:
            if fig3 is None or ax3 is None:
                fig3, ax3 = plt.subplots(1, 1, figsize=(7, 6))
            fig3 = self._hr_heatmap_gaia(fig3, ax3, mask_parallax, cmap)
        return ax1, fig2, axs2
